#!/usr/bin/env python3
"""
Convert a directory of georeferenced tiles and per-tile GeoJSON labels into COCO files.

Expected layout (customisable with CLI flags):

dataset_dir/
  train/
    RGB/
      <tile>.tif
    labels/
      <tile>.geojson
  val/
  test/

Each GeoJSON file must share the same stem as the tile and contain Polygon/MultiPolygon
geometries describing crowns in the same CRS as the raster.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
from PIL import Image
from affine import Affine
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
import yaml


def inv(transform: Affine) -> Affine:
    return ~transform


def to_px_coords(poly: Polygon, A_inv: Affine) -> List[float]:
    x, y = poly.exterior.coords.xy
    cols, rows = A_inv * (np.asarray(x), np.asarray(y))
    seg: List[float] = []
    for c, r in zip(cols, rows):
        seg += [float(c), float(r)]
    return seg


def bbox_xywh(seg: List[float]) -> List[float]:
    xs, ys = seg[0::2], seg[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]


@dataclass
class TileResult:
    tile_path: str
    label_path: str
    width: int = 0
    height: int = 0
    annotations: List[Dict[str, object]] | None = None
    reason: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.annotations is None:
            self.annotations = []


def area_in_px(poly: Polygon, A_inv: Affine) -> float:
    x, y = poly.exterior.coords.xy
    cols, rows = A_inv * (np.asarray(x), np.asarray(y))
    poly_px = Polygon([(float(c), float(r)) for c, r in zip(cols, rows)])
    return float(abs(poly_px.area))


def read_tile_metadata(tile_path: Path) -> Tuple[int, int, Affine]:
    with Image.open(tile_path) as img:
        width, height = img.size
        tags = img.tag_v2
        if 34264 in tags:
            matrix = tuple(float(v) for v in tags[34264])
            if len(matrix) != 16:
                raise ValueError(f"Invalid ModelTransformationTag for {tile_path}")
            a, b, c, d, e, f, g, h, *_ = matrix
            transform = Affine.from_gdal(d, a, b, h, e, f)
        elif 33550 in tags and 33922 in tags:
            scale = tags[33550]
            tiepoints = tags[33922]
            if len(tiepoints) < 6:
                raise ValueError(f"Invalid ModelTiepointTag for {tile_path}")
            scale_x = float(scale[0])
            scale_y = float(scale[1])
            origin_x = float(tiepoints[3])
            origin_y = float(tiepoints[4])
            transform = Affine(scale_x, 0.0, origin_x, 0.0, -scale_y, origin_y)
        else:
            raise ValueError(f"Missing GeoTIFF tags for georeferencing in {tile_path}")
    return width, height, transform


def compute_tile_bounds(transform: Affine, width: int, height: int) -> Tuple[float, float, float, float]:
    corners = [
        transform * (0, 0),
        transform * (width, 0),
        transform * (width, height),
        transform * (0, height),
    ]
    xs, ys = zip(*corners)
    return min(xs), min(ys), max(xs), max(ys)


def read_config(path: Path | None) -> Dict[str, object]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration file must contain a mapping")
    return raw


def resolve_paths_and_splits(
    args: argparse.Namespace, config: Dict[str, object]
) -> Tuple[Path, Path, List[str]]:
    cfg_paths = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    cfg_processing = config.get("processing", {}) if isinstance(config.get("processing"), dict) else {}

    dataset_dir: Path | None = args.dataset_dir
    if dataset_dir is None:
        default_dir = cfg_paths.get("output_dir")
        if default_dir is not None:
            dataset_dir = Path(default_dir)
    if dataset_dir is None:
        raise ValueError("Provide --dataset_dir or set paths.output_dir inside the config file")
    dataset_dir = dataset_dir.expanduser().resolve()

    out_dir: Path | None = args.out_dir
    if out_dir is None:
        coco_dir = cfg_paths.get("coco_dir")
        if coco_dir is not None:
            out_dir = Path(coco_dir)
        else:
            out_dir = dataset_dir.parent / "coco"
    out_dir = out_dir.expanduser().resolve()

    splits = args.splits
    if splits is None:
        cfg_splits = cfg_processing.get("splits") if isinstance(cfg_processing.get("splits"), (list, tuple)) else None
        if cfg_splits:
            splits = [str(split) for split in cfg_splits]
        else:
            splits = ["train", "val", "test"]
    splits = [split.lower() for split in splits]
    return dataset_dir, out_dir, splits


def process_tile(
    task: Tuple[str, str],
    category_field: str,
    fallback_fields: Sequence[str],
    value_map: Dict[str, str] | None,
    map_default: str | None,
) -> TileResult:
    tile_path_str, label_path_str = task
    tile_path = Path(tile_path_str)
    label_path = Path(label_path_str)
    try:
        records = load_tile_labels(
            label_path,
            category_field,
            fallback_fields=fallback_fields,
            value_map=value_map,
            map_default=map_default,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        return TileResult(tile_path_str, label_path_str, reason="label_error", error=str(exc))

    if not records:
        return TileResult(tile_path_str, label_path_str, reason="empty_labels")

    try:
        width, height, transform = read_tile_metadata(tile_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        return TileResult(tile_path_str, label_path_str, reason="metadata_error", error=str(exc))

    tile_poly = box(*compute_tile_bounds(transform, width, height))
    A_inv = inv(transform)
    annotations: List[Dict[str, object]] = []
    for category, geom in records:
        clipped = geom.intersection(tile_poly)
        if clipped.is_empty:
            continue
        for poly in flatten_polygons(clipped):
            if poly.is_empty or poly.area == 0:
                continue
            seg = to_px_coords(poly, A_inv)
            if len(seg) < 6:
                continue
            annotations.append(
                {
                    "id": 0,
                    "image_id": 0,
                    "category_name": category,
                    "segmentation": [seg],
                    "bbox": bbox_xywh(seg),
                    "area": area_in_px(poly, A_inv),
                    "iscrowd": 0,
                }
            )

    if not annotations:
        return TileResult(tile_path_str, label_path_str, width=width, height=height, reason="no_annotations")

    return TileResult(tile_path_str, label_path_str, width=width, height=height, annotations=annotations)


def run_tile_jobs(
    tasks: List[Tuple[str, str]],
    *,
    workers: int,
    category_field: str,
    fallback_fields: Sequence[str],
    value_map: Dict[str, str] | None,
    map_default: str | None,
) -> List[TileResult]:
    if not tasks:
        return []
    fallback_fields = tuple(fallback_fields)
    worker_count = workers
    if worker_count <= 0:
        cpu = os.cpu_count() or 1
        worker_count = min(cpu, len(tasks))
    else:
        worker_count = min(worker_count, len(tasks))

    if worker_count <= 1:
        return [
            process_tile(task, category_field, fallback_fields, value_map, map_default)
            for task in tasks
        ]

    with cf.ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            executor.map(
                process_tile,
                tasks,
                repeat(category_field),
                repeat(fallback_fields),
                repeat(value_map),
                repeat(map_default),
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config (same format as create_dataset.py) used to infer paths and splits.",
    )
    ap.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="Root directory that contains <split>/<tiles_subdir> and <split>/<labels_subdir> folders.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory where COCO JSON files will be written. Defaults to <dataset_dir>/../coco if omitted.",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="List of dataset splits to process. Defaults to config.processing.splits or train/val/test.",
    )
    ap.add_argument("--tiles_subdir", default="RGB", help="Name of the subdirectory that stores raster tiles.")
    ap.add_argument("--labels_subdir", default="labels", help="Name of the subdirectory that stores GeoJSON labels.")
    ap.add_argument(
        "--category_field",
        default="class_code",
        help="Property inside each GeoJSON feature used to derive the COCO category name.",
    )
    ap.add_argument(
        "--fallback_fields",
        nargs="*",
        default=["scientific_name", "class_code"],
        help="If the main category field is missing/empty, try these fields in order.",
    )
    ap.add_argument(
        "--map_json",
        type=Path,
        default=None,
        help="Optional JSON file mapping raw label values to desired category names (case-insensitive keys).",
    )
    ap.add_argument(
        "--map_default",
        type=str,
        default=None,
        help="Default category to use when a value is not found in --map_json. If omitted, unknowns are skipped.",
    )
    ap.add_argument(
        "--image_exts",
        nargs="+",
        default=[".tif", ".tiff"],
        help="Image extensions to consider when scanning the tiles directory.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0=auto, 1=sequential).",
    )
    return ap.parse_args()


def list_tiles(tiles_dir: Path, image_exts: Sequence[str]) -> List[Path]:
    wanted = {ext if ext.startswith(".") else f".{ext}" for ext in (ext.lower() for ext in image_exts)}
    tiles: List[Path] = []
    for path in tiles_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in wanted:
            tiles.append(path)
    return sorted(tiles)


def load_tile_labels(label_path: Path, category_field: str, *,
                     fallback_fields: Sequence[str] = (),
                     value_map: Dict[str, str] | None = None,
                     map_default: str | None = None) -> List[Tuple[str, Polygon]]:
    with label_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    features = data.get("features", [])
    records: List[Tuple[str, Polygon]] = []
    for feat in features:
        geom = feat.get("geometry")
        props = feat.get("properties") or {}
        if not geom:
            continue
        val: str = ""
        if category_field in props and props[category_field]:
            val = str(props[category_field]).strip()
        if not val:
            for fb in fallback_fields:
                if fb in props and props[fb]:
                    val = str(props[fb]).strip()
                    if val:
                        break
        if not val:
            continue
        if value_map is not None:
            key = val.lower()
            if key in value_map:
                val = value_map[key]
            elif map_default is not None:
                val = map_default
            else:
                # skip unknowns when no default provided
                continue
        category = val
        try:
            shp = shape(geom)
        except Exception:
            continue
        shp = make_valid(shp)
        if shp.is_empty:
            continue
        records.append((category, shp))
    return records


def flatten_polygons(geom) -> Iterable[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if not poly.is_empty]
    if isinstance(geom, GeometryCollection):
        polys: List[Polygon] = []
        for part in geom.geoms:
            polys.extend(flatten_polygons(part))
        return polys
    return []


def build_categories(class_names: Iterable[str]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    sorted_names = sorted(class_names)
    class_map = {name: idx + 1 for idx, name in enumerate(sorted_names)}
    categories = [{"id": cid, "name": name, "supercategory": "tree"} for name, cid in class_map.items()]
    return categories, class_map


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    dataset_dir, out_dir, splits = resolve_paths_and_splits(args, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        split: {"images": [], "annotations": [], "categories": []} for split in splits
    }
    next_image_id = {split: 1 for split in splits}
    next_ann_id = {split: 1 for split in splits}
    class_names: set[str] = set()

    # Optional mapping for category normalization
    value_map: Dict[str, str] | None = None
    if args.map_json is not None:
        with args.map_json.open("r", encoding="utf-8") as h:
            raw_map = json.load(h)
        value_map = {str(k).lower(): str(v) for k, v in raw_map.items()}

    missing_labels: List[Path] = []
    empty_labels: List[Path] = []
    tiles_without_annotations = 0
    processing_errors: List[Tuple[Path, str]] = []

    for split in splits:
        tiles_dir = dataset_dir / split / args.tiles_subdir
        labels_dir = dataset_dir / split / args.labels_subdir
        if not tiles_dir.is_dir():
            raise FileNotFoundError(f"Tiles directory missing: {tiles_dir}")
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Labels directory missing: {labels_dir}")

        tile_paths = list_tiles(tiles_dir, args.image_exts)
        tasks: List[Tuple[str, str]] = []
        for tile_path in tile_paths:
            label_path = labels_dir / (tile_path.stem + ".geojson")
            if not label_path.is_file():
                missing_labels.append(label_path)
                continue
            tasks.append((str(tile_path), str(label_path)))

        results = run_tile_jobs(
            tasks,
            workers=args.workers,
            category_field=args.category_field,
            fallback_fields=args.fallback_fields,
            value_map=value_map,
            map_default=args.map_default,
        )

        for result in results:
            tile_path = Path(result.tile_path)
            label_path = Path(result.label_path)
            if result.error:
                processing_errors.append((label_path, result.error))
                continue
            if result.reason == "empty_labels":
                empty_labels.append(label_path)
                continue
            if result.reason == "no_annotations":
                tiles_without_annotations += 1
                continue
            rel_name = (Path(split) / args.tiles_subdir / tile_path.name).as_posix()
            image_id = next_image_id[split]
            coco[split]["images"].append({"id": image_id, "file_name": rel_name, "width": result.width, "height": result.height})
            for ann in result.annotations:
                ann["id"] = next_ann_id[split]
                ann["image_id"] = image_id
                coco[split]["annotations"].append(ann)
                class_names.add(ann["category_name"])
                next_ann_id[split] += 1
            next_image_id[split] += 1

    if not class_names:
        raise ValueError("No categories found. Check that category_field is correct and labels contain features.")

    categories, class_map = build_categories(class_names)
    for split in splits:
        coco[split]["categories"] = categories
        for ann in coco[split]["annotations"]:
            cname = ann.pop("category_name")
            ann["category_id"] = class_map[cname]

    for split in splits:
        out_path = out_dir / f"{split}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(coco[split], handle, indent=2)
        print(
            f"Wrote {out_path} "
            f"({len(coco[split]['images'])} images, {len(coco[split]['annotations'])} annotations, "
            f"{len(categories)} categories)"
        )

    if missing_labels:
        print(f"Warning: {len(missing_labels)} tiles skipped because label files were missing.")
    if empty_labels:
        print(f"Warning: {len(empty_labels)} label files contained no usable features.")
    if tiles_without_annotations:
        print(f"Warning: {tiles_without_annotations} tiles produced no annotations after clipping.")
    if processing_errors:
        sample = processing_errors[0]
        print(
            "Warning: "
            f"{len(processing_errors)} label files failed during processing. "
            f"First failure: {sample[0]} -> {sample[1]}"
        )


if __name__ == "__main__":
    main()
