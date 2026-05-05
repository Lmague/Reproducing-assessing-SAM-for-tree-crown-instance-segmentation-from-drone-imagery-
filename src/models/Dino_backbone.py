import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

# --- 1. OUTILS D'ARCHITECTURE ---

class DSMEncoder(nn.Module):
    """
    Petit CNN pour traiter le DSM (1 canal) et l'amener à la résolution de DINO.
    """
    def __init__(self, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),   # /2
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),  # /4
            nn.Conv2d(64, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels), nn.ReLU() # /8
        )

    def forward(self, x):
        return self.net(x)

class SimpleFeaturePyramid(nn.Module):
    """
    Adaptateur ViTDet : Transforme la sortie unique de DINO en pyramide FPN (P2-P5).
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        # P2 (1/4)
        self.p2_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
        # P3 (1/8)
        self.p3_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )
        # P4 (1/16) - Échelle native de DINO
        self.p4_block = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # P5 (1/32)
        self.p5_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return [self.p2_block(x), self.p3_block(x), self.p4_block(x), self.p5_block(x)]

# --- 2. LE BACKBONE PRINCIPAL ---

@BACKBONE_REGISTRY.register()
class DINO_DSM_Backbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        print("🦖 Initialisation du Backbone DINOv3 + DSM...")
        
        # A. Chargement de DINO (Méthode compatible DINOv2/v3)
        # Utilisation de torch.hub pour avoir l'architecture ViT propre
        # Pour le 7B, remplace 'dinov2_vitl14' par 'dinov2_vitg14' si disponible ou l'équivalent
        base_model_name = 'dinov2_vitl14' 
        
        self.dino = torch.hub.load('facebookresearch/dinov2', base_model_name)
        
        # --- CHARGEMENT DES POIDS SATELLITE (OPTIONNEL) ---
        # Si tu as ton fichier .pth téléchargé (ex: ViT-L_16_distilled_sat.pth)
        # weights_path = "./ViT-L_16_distilled_sat.pth"
        # if os.path.exists(weights_path):
        #     print(f"Chargement des poids satellite depuis {weights_path}")
        #     state_dict = torch.load(weights_path, map_location="cpu")
        #     self.dino.load_state_dict(state_dict, strict=False)
        
        # B. Optimisation L40S (bfloat16)
        # bfloat16 est plus stable que float16 pour les gros modèles
        self.dino.to(torch.bfloat16) 
        self.dino.eval() # Toujours gelé
        for p in self.dino.parameters(): 
            p.requires_grad = False
            
        dino_dim = self.dino.embed_dim
        print(f"Dimensions DINO : {dino_dim}")

        # C. Module DSM
        dsm_dim = 64
        self.dsm_encoder = DSMEncoder(out_channels=dsm_dim)
        
        # D. Fusion & Pyramide
        self.fusion_conv = nn.Conv2d(dino_dim + dsm_dim, 256, kernel_size=1)
        self.fpn = SimpleFeaturePyramid(in_channels=256, out_channels=256)
        
        # Métadonnées Detectron2
        self._out_features = ["p2", "p3", "p4", "p5"]
        self._out_feature_strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
        self._out_feature_channels = {k: 256 for k in self._out_features}

    def forward(self, x):
        # x est le tenseur (B, 4, H, W) fourni par ton Mapper 4 canaux
        rgb = x[:, :3]
        dsm = x[:, 3:]
        
        B, _, H, W = rgb.shape
        
        # 1. Préparation DINO (Padding patch 14)
        h_pad = (14 - H % 14) % 14
        w_pad = (14 - W % 14) % 14
        # Passage en bfloat16 juste pour DINO
        rgb_pad = F.pad(rgb, (0, w_pad, 0, h_pad)).to(torch.bfloat16)
        
        with torch.no_grad():
            # Forward DINO
            out = self.dino.forward_features(rgb_pad)
            tokens = out["x_norm_patchtokens"]
            
            # Reshape (B, N, D) -> (B, D, H, W)
            Hp, Wp = (H + h_pad) // 14, (W + w_pad) // 14
            dino_feat = tokens.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).float() # Retour en float32
            
            # Crop si nécessaire pour enlever le padding (souvent inutile si fusion gère le resize)

        # 2. Traitement DSM
        dsm_feat = self.dsm_encoder(dsm) # Sortie stride 8
        # Alignement sur la taille DINO (stride 14 approx)
        dsm_feat = F.interpolate(dsm_feat, size=dino_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        # 3. Fusion
        fused = torch.cat([dino_feat, dsm_feat], dim=1)
        fused = self.fusion_conv(fused)
        
        # 4. Pyramide
        # On s'assure d'être sur un stride 16 propre pour le FPN
        target_h, target_w = H // 16, W // 16
        fused_16 = F.interpolate(fused, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        features = self.fpn(fused_16)
        
        return {
            "p2": features[0],
            "p3": features[1],
            "p4": features[2],
            "p5": features[3]
        }

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }