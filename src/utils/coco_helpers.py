"""
Shared COCO JSON utilities.

Used by training scripts to fix missing fields in COCO annotation files
before loading them with pycocotools.
"""
import json


def fix_coco_json_if_needed(json_path):
    """
    Add missing 'info' and 'licenses' fields to a COCO JSON file if needed.
    Fixes the KeyError: 'info' from pycocotools.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    modified = False

    if 'info' not in data:
        data['info'] = {
            "description": "Tree Crown Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": ""
        }
        modified = True

    if 'licenses' not in data:
        data['licenses'] = []
        modified = True

    if modified:
        with open(json_path, 'w') as f:
            json.dump(data, f)
        print(f"Fixed COCO JSON: {json_path}")
