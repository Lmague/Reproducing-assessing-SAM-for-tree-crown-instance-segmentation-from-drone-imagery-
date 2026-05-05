#!/usr/bin/env python3
"""
Compute average tree/object size statistics from a COCO dataset (val or test).

Works with:
- bbox sizes (always)
- annotation "area" (preferred if present)
- per-category stats
- relative sizes vs image size

Usage:
  python coco_tree_size_stats.py --json /path/to/val.json
  python coco_tree_size_stats.py --json /path/to/test.json --topk 20
"""

import argparse
import json
import math
import statistics
from collections import defaultdict

import numpy as np


def pct(values, p):
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def safe_mean(xs):
    return float(np.mean(xs)) if xs else None


def safe_std(xs):
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else None


def safe_median(xs):
    return float(np.median(xs)) if xs else None


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def markdown_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join([":---" for _ in headers]) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def compute_stats(items):
    # items: dict[str, list[float]]
    # returns dict[str, dict[str, float]]
    stats = {}
    for k, xs in items.items():
        xs = [float(v) for v in xs if v is not None and not math.isnan(v)]
        stats[k] = {
            "n": len(xs),
            "mean": safe_mean(xs),
            "std": safe_std(xs),
            "median": safe_median(xs),
            "p10": pct(xs, 10),
            "p25": pct(xs, 25),
            "p75": pct(xs, 75),
            "p90": pct(xs, 90),
            "min": float(min(xs)) if xs else None,
            "max": float(max(xs)) if xs else None,
        }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to COCO annotations json (val.json or test.json)")
    ap.add_argument("--topk", type=int, default=10, help="Top-K categories to print by count")
    ap.add_argument("--include_crowd", action="store_true", help="Include iscrowd=1 annotations")
    args = ap.parse_args()

    with open(args.json, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    img_by_id = {im["id"]: im for im in images if "id" in im}
    cat_by_id = {c["id"]: c.get("name", str(c["id"])) for c in cats if "id" in c}

    # Global lists
    glob = defaultdict(list)          # metric -> list
    per_cat = defaultdict(lambda: defaultdict(list))  # cat_name -> metric -> list
    counts = defaultdict(int)

    skipped_no_img = 0
    skipped_crowd = 0

    for ann in anns:
        if not args.include_crowd and ann.get("iscrowd", 0) == 1:
            skipped_crowd += 1
            continue

        img = img_by_id.get(ann.get("image_id"))
        if img is None:
            skipped_no_img += 1
            continue

        iw = float(img.get("width", 0) or 0)
        ih = float(img.get("height", 0) or 0)
        if iw <= 0 or ih <= 0:
            skipped_no_img += 1
            continue

        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue

        _, _, bw, bh = bbox
        bw = float(bw)
        bh = float(bh)
        bbox_area = bw * bh

        # Prefer COCO "area" if present; else fallback bbox area
        area = float(ann.get("area", bbox_area))

        # “Equivalent diameter” of a circle with same area (in px)
        eq_diam = 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0

        # Relative measures
        rel_area = area / (iw * ih)
        rel_bw = bw / iw
        rel_bh = bh / ih

        cat_id = ann.get("category_id", None)
        cat_name = cat_by_id.get(cat_id, str(cat_id))

        counts[cat_name] += 1

        # Store global
        glob["bbox_w_px"].append(bw)
        glob["bbox_h_px"].append(bh)
        glob["bbox_area_px2"].append(bbox_area)
        glob["ann_area_px2"].append(area)
        glob["eq_diam_px"].append(eq_diam)
        glob["rel_area"].append(rel_area)
        glob["rel_bbox_w"].append(rel_bw)
        glob["rel_bbox_h"].append(rel_bh)

        # Store per-cat
        per_cat[cat_name]["bbox_w_px"].append(bw)
        per_cat[cat_name]["bbox_h_px"].append(bh)
        per_cat[cat_name]["bbox_area_px2"].append(bbox_area)
        per_cat[cat_name]["ann_area_px2"].append(area)
        per_cat[cat_name]["eq_diam_px"].append(eq_diam)
        per_cat[cat_name]["rel_area"].append(rel_area)

    # Compute stats
    gstats = compute_stats(glob)

    # Print summary
    print("\n# COCO size stats")
    print(f"- JSON: {args.json}")
    print(f"- images: {len(images)}")
    print(f"- annotations total: {len(anns)}")
    print(f"- used annotations: {gstats['ann_area_px2']['n']}")
    print(f"- skipped (no image dims): {skipped_no_img}")
    print(f"- skipped (iscrowd): {skipped_crowd}  (use --include_crowd to include)\n")

    # Global table
    headers = ["metric", "n", "mean", "std", "median", "p10", "p25", "p75", "p90", "min", "max"]
    rows = []
    for m in ["ann_area_px2", "eq_diam_px", "bbox_w_px", "bbox_h_px", "rel_area", "rel_bbox_w", "rel_bbox_h"]:
        s = gstats[m]
        rows.append([
            m,
            s["n"],
            fmt(s["mean"], 4),
            fmt(s["std"], 4),
            fmt(s["median"], 4),
            fmt(s["p10"], 4),
            fmt(s["p25"], 4),
            fmt(s["p75"], 4),
            fmt(s["p90"], 4),
            fmt(s["min"], 4),
            fmt(s["max"], 4),
        ])
    print("## Global")
    print(markdown_table(headers, rows))

    # Per-category topK by count
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: args.topk]
    print(f"\n## Per-category (top {args.topk} by count)")
    cheaders = ["category", "n", "area_mean(px2)", "area_med(px2)", "diam_mean(px)", "diam_med(px)", "rel_area_mean"]
    crows = []
    for cat_name, n in top:
        s_area = compute_stats({"a": per_cat[cat_name]["ann_area_px2"]})["a"]
        s_d = compute_stats({"d": per_cat[cat_name]["eq_diam_px"]})["d"]
        s_ra = compute_stats({"r": per_cat[cat_name]["rel_area"]})["r"]
        crows.append([
            cat_name,
            n,
            fmt(s_area["mean"], 2),
            fmt(s_area["median"], 2),
            fmt(s_d["mean"], 2),
            fmt(s_d["median"], 2),
            fmt(s_ra["mean"], 6),
        ])
    print(markdown_table(cheaders, crows))

    print("\nNotes:")
    print("- ann_area_px2 uses COCO annotation 'area' when present; else bbox_w*bbox_h.")
    print("- eq_diam_px is the diameter of a circle with the same area (helps intuition).")
    print("- rel_area is area / (image_width*image_height).")


if __name__ == "__main__":
    main()
