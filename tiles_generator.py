#!/usr/bin/env python3
"""
Générateur de Jeu de Données en Tuiles au format COCO pour la Segmentation de Cimes d'Arbres.

Ce script prend un jeu de données géospatial (rasters RGB/DSM et polygones vectoriels
avec des splits pré-définis) et le convertit en un jeu de données de segmentation d'instances
en tuiles, formaté selon les standards COCO, en suivant la méthodologie de l'article
"Assessing SAM for Tree Crown Instance Segmentation from Drone Imagery".
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from sahi.annotation import Annotation
from sahi.slicing import slice_image
from shapely.geometry import box
from tqdm import tqdm


def configure_logging():
    """Initialise le logging pour la console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def find_raster(data_dir: Path, site: str, kind: str) -> Path | None:
    """Localise le raster correspondant au site et au type ('rgb' ou 'dsm')."""
    sub_dir = data_dir / kind.upper()
    if not sub_dir.is_dir():
        logging.warning(f"Le sous-dossier '{sub_dir}' est introuvable.")
        return None
    
    suffixes = (".tif", ".tiff")
    for suffix in suffixes:
        candidates = sorted(sub_dir.glob(f"*{site}*{suffix}"))
        if candidates:
            return candidates[0]
    logging.warning(f"Aucun raster {kind.upper()} trouvé pour le site '{site}' dans {sub_dir}")
    return None

def export_to_coco(coco_data: Dict[str, List], class_map: Dict[str, int], out_dir: Path):
    """Exporte les données collectées au format COCO, avec un fichier JSON par split."""
    logging.info("Exportation du jeu de données au format COCO...")
    
    categories = [{"id": v, "name": k, "supercategory": "tree"} for k, v in class_map.items()]
    
    for split in ["train", "val", "test"]:
        split_data = coco_data.get(split)
        if not split_data:
            logging.info(f"Aucune donnée pour le split '{split}', fichier JSON non généré.")
            continue

        images = []
        annotations = []
        annotation_id_counter = 1

        for image_id, (image_info, ann_info_list) in enumerate(split_data, 1):
            image_info["id"] = image_id
            images.append(image_info)
            
            for ann_info in ann_info_list:
                ann_info["id"] = annotation_id_counter
                ann_info["image_id"] = image_id
                annotations.append(ann_info)
                annotation_id_counter += 1

        final_coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        output_path = out_dir / f"annotations_{split}.json"
        with output_path.open("w") as f:
            json.dump(final_coco, f, indent=4)
        logging.info(f"Fichier COCO pour le split '{split}' sauvegardé dans {output_path}")

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path, help="Répertoire racine contenant les sous-dossiers RGB et DSM.")
    parser.add_argument("--labels_gpkg", required=True, type=Path, help="GeoPackage contenant les polygones des arbres et la colonne 'split'.")
    parser.add_argument("--aoi_gpkg", required=True, type=Path, help="GeoPackage contenant les polygones des Zones d'Intérêt (AOI).")
    parser.add_argument("--out_dir", required=True, type=Path, help="Répertoire de sortie pour le jeu de données COCO.")
    parser.add_argument("--tile_size", type=int, default=1024, help="Taille des tuiles (carrées).")
    parser.add_argument("--overlap_ratio", type=float, default=0.5, help="Ratio de chevauchement entre les tuiles.")
    parser.add_argument("--min_area_ratio", type=float, default=0.2, help="Ratio minimum de surface d'une annotation pour être conservée dans une tuile.")
    parser.add_argument("--max_nodata_ratio", type=float, default=0.8, help="Ratio maximum de pixels hors AOI (no-data) autorisé dans une tuile.")
    args = parser.parse_args()
    
    configure_logging()

    # --- PHASE 1 : INITIALISATION ET CHARGEMENT ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = args.out_dir / "images"
    dsm_out_dir = args.out_dir / "dsm"
    images_out_dir.mkdir(exist_ok=True)
    dsm_out_dir.mkdir(exist_ok=True)

    logging.info(f"Chargement des labels depuis {args.labels_gpkg}...")
    labels_gdf = gpd.read_file(args.labels_gpkg)
    logging.info(f"Chargement des AOIs depuis {args.aoi_gpkg}...")
    aois_gdf = gpd.read_file(args.aoi_gpkg)

    # Créer la map des catégories pour COCO
    species = sorted(labels_gdf["species"].unique())
    class_map = {name: i + 1 for i, name in enumerate(species)}
    logging.info(f"Classes détectées : {class_map}")

    sites_to_process = sorted(labels_gdf["site"].unique())
    coco_data = {"train": [], "val": [], "test": []}

    # --- PHASE 2 : TILING ET COLLECTE DES DONNÉES ---
    for site in tqdm(sites_to_process, desc="Traitement des sites"):
        rgb_path = find_raster(args.data_dir, site, "rgb")
        dsm_path = find_raster(args.data_dir, site, "dsm")
        if not rgb_path or not dsm_path:
            continue
            
        with rasterio.open(rgb_path) as src_rgb, rasterio.open(dsm_path) as src_dsm:
            # Préparer les annotations pour Sahi
            site_labels = labels_gdf[labels_gdf["site"] == site]
            sahi_annotations = []
            for row in site_labels.itertuples():
                poly = row.geometry
                bbox = [int(p) for p in poly.bounds] # [minx, miny, maxx, maxy]
                sahi_annotations.append(Annotation(
                    bbox=bbox,
                    segmentation=np.array(poly.exterior.coords).flatten().tolist(),
                    category_name=row.species_final,
                    category_id=class_map[row.species_final],
                    full_shape=row.geometry, # Stocker la géométrie complète pour le calcul de ratio
                    split=row.split
                ))

            # Créer le masque AOI pour le site entier
            site_aois = aois_gdf[aois_gdf["site"] == site]
            aoi_mask = rasterio.features.rasterize(
                site_aois.geometry,
                out_shape=src_rgb.shape,
                transform=src_rgb.transform,
                fill=0,
                dtype="uint8"
            )

            # Slicing avec Sahi
            slice_result = slice_image(
                image=str(rgb_path),
                annotations=sahi_annotations,
                slice_height=args.tile_size,
                slice_width=args.tile_size,
                overlap_height_ratio=args.overlap_ratio,
                overlap_width_ratio=args.overlap_ratio,
                min_area_ratio=args.min_area_ratio,
                verbose=False
            )

            for sliced_image in slice_result["sliced_images"]:
                # --- PHASE 3 : FILTRAGE ET COLLECTE DES ANNOTATIONS ---
                if not sliced_image.annotations:
                    continue

                # Filtrer les tuiles avec trop de pixels hors AOI
                x, y, w, h = sliced_image.starting_pixel
                aoi_tile_mask = aoi_mask[y:y+h, x:x+w]
                valid_pixel_ratio = aoi_tile_mask.mean()
                if 1 - valid_pixel_ratio > args.max_nodata_ratio:
                    continue

                # Sauvegarder la tuile RGB
                tile_filename = f"{site}_{y}_{x}.png"
                Image.fromarray(sliced_image.image).save(images_out_dir / tile_filename)
                
                # Sauvegarder la tuile DSM correspondante
                dsm_tile_filename = f"{site}_{y}_{x}.tif"
                dsm_window = Window(x, y, w, h)
                dsm_tile_data = src_dsm.read(1, window=dsm_window)
                dsm_transform = src_dsm.window_transform(dsm_window)
                with rasterio.open(
                    dsm_out_dir / dsm_tile_filename, 'w', driver='GTiff', 
                    height=h, width=w, count=1, dtype=dsm_tile_data.dtype, 
                    crs=src_dsm.crs, transform=dsm_transform
                ) as dst:
                    dst.write(dsm_tile_data, 1)

                # Collecter les données pour COCO
                image_info = {"file_name": tile_filename, "height": h, "width": w}
                ann_info_list = []
                for ann in sliced_image.annotations:
                    ann_info_list.append({
                        "category_id": ann.category_id,
                        "segmentation": [ann.segmentation],
                        "bbox": ann.bbox.to_xywh(),
                        "area": ann.area,
                        "iscrowd": 0,
                    })

                # Déterminer le split de la tuile (majorité des annotations)
                tile_split = Counter(ann.split for ann in sliced_image.annotations).most_common(1)[0][0]
                coco_data[tile_split].append((image_info, ann_info_list))

    # --- PHASE 4 : EXPORT FINAL AU FORMAT COCO ---
    export_to_coco(coco_data, class_map, args.out_dir)
    logging.info("Jeu de données au format COCO généré avec succès.")

if __name__ == "__main__":
    main()