#!/bin/bash
# =============================================================================
# Setup + Launch Mask2Former Swin-T pour tree crown instance segmentation
#
# Ce script:
# 1. Clone le repo Mask2Former (Facebook Research)
# 2. Compile l'extension CUDA MSDeformAttn
# 3. Télécharge le checkpoint COCO pretrained Swin-T
# 4. Lance l'entraînement
#
# Usage:
#   bash src/training/setup_mask2former.sh           # Setup + train
#   bash src/training/setup_mask2former.sh --skip-setup  # Train only (déjà installé)
# =============================================================================

set -e

WORKDIR="/workspace/Documents/tree-segmentation"
MASK2FORMER_DIR="/workspace/Mask2Former"
LOG="/workspace/training_logs/mask2former.log"
VENV="/venv/main/bin/activate"

# Activate venv
source $VENV 2>/dev/null || true

echo "============================================================"
echo "  Mask2Former Swin-T — Setup & Training"
echo "============================================================"
echo "  Date: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "============================================================"

# =============================================================================
# STEP 0: Check if --skip-setup flag
# =============================================================================
SKIP_SETUP=false
BATCH_SIZE=2
for arg in "$@"; do
    case $arg in
        --skip-setup) SKIP_SETUP=true ;;
        --batch-size=*) BATCH_SIZE="${arg#*=}" ;;
    esac
done

# =============================================================================
# STEP 1: Clone Mask2Former
# =============================================================================
if [ "$SKIP_SETUP" = false ]; then
    echo ""
    echo "📦 Étape 1/4: Clone Mask2Former..."

    if [ ! -d "$MASK2FORMER_DIR" ]; then
        cd /workspace
        git clone https://github.com/facebookresearch/Mask2Former.git
        echo "  ✅ Cloné"
    else
        echo "  ✅ Déjà cloné"
    fi

    # =============================================================================
    # STEP 2: Install dependencies
    # =============================================================================
    echo ""
    echo "📦 Étape 2/4: Installation des dépendances..."

    pip install -q timm scipy shapely h5py submitit 2>&1 | tail -1

    # =============================================================================
    # STEP 3: Build MSDeformAttn CUDA extension
    # =============================================================================
    echo ""
    echo "🔨 Étape 3/4: Compilation MSDeformAttn..."

    MSDA_DIR="$MASK2FORMER_DIR/mask2former/modeling/pixel_decoder/ops"
    if [ -d "$MSDA_DIR" ]; then
        cd "$MSDA_DIR"

        # Check if already built
        if python3 -c "from MultiScaleDeformableAttention import MultiScaleDeformableAttention" 2>/dev/null; then
            echo "  ✅ Déjà compilé"
        else
            echo "  Compilation en cours (peut prendre 2-5 min)..."
            # Set CUDA arch for RTX 3090 (SM 8.6)
            export TORCH_CUDA_ARCH_LIST="8.6"
            python3 setup.py build_ext --inplace 2>&1 | tail -5
            if python3 -c "from MultiScaleDeformableAttention import MultiScaleDeformableAttention" 2>/dev/null; then
                echo "  ✅ Compilation réussie"
            else
                echo "  ❌ Compilation échouée — tentative d'installation pip..."
                pip install MultiScaleDeformableAttention 2>&1 | tail -3 || true
                if python3 -c "from MultiScaleDeformableAttention import MultiScaleDeformableAttention" 2>/dev/null; then
                    echo "  ✅ Installé via pip"
                else
                    echo "  ❌ ERREUR: MSDeformAttn non disponible. Abandon."
                    exit 1
                fi
            fi
        fi
    else
        echo "  ❌ Répertoire MSDeformAttn introuvable: $MSDA_DIR"
        exit 1
    fi

    # =============================================================================
    # STEP 4: Download COCO pretrained checkpoint
    # =============================================================================
    echo ""
    echo "📥 Étape 4/4: Vérification du checkpoint COCO pretrained..."

    CKPT_DIR="/workspace/checkpoints"
    CKPT_FILE="$CKPT_DIR/mask2former_swin_tiny_coco_instance.pkl"
    CKPT_URL="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"

    mkdir -p $CKPT_DIR
    if [ ! -f "$CKPT_FILE" ]; then
        echo "  Téléchargement (~300 MB)..."
        wget -q --show-progress -O "$CKPT_FILE" "$CKPT_URL"
        echo "  ✅ Téléchargé: $CKPT_FILE"
    else
        echo "  ✅ Déjà téléchargé"
    fi
fi

# =============================================================================
# STEP 5: Launch training
# =============================================================================
echo ""
echo "🚀 Lancement de l'entraînement Mask2Former Swin-T..."
echo "   Batch size: $BATCH_SIZE"
echo "   Log: $LOG"
echo "============================================================"

cd $WORKDIR
mkdir -p /workspace/training_logs

# Add Mask2Former to PYTHONPATH
export PYTHONPATH="${MASK2FORMER_DIR}:${WORKDIR}:${PYTHONPATH}"

python3 src/training/train_mask2former.py \
    --batch-size $BATCH_SIZE \
    2>&1 | tee $LOG

echo ""
echo "============================================================"
echo "  ✅ Entraînement terminé!"
echo "  Log: $LOG"
echo "  Output: $WORKDIR/src/training/Mask2Former_SwinT_Trees/"
echo "============================================================"
