"""
dataset_load_smoke.py
Runs sanity checks for COCO and VQAv2 dataset loaders.
"""

import sys, os, pathlib, textwrap
from pprint import pprint

# make sure repo root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from datasets.coco_loader import load_coco_small
from datasets.vqa_loader import load_vqa_small


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_record(record: dict, keys=("image", "caption", "question", "answers")):
    summary = {}
    for k in keys:
        if k in record:
            val = record[k]
            if isinstance(val, str):
                summary[k] = textwrap.shorten(val, width=70)
            elif isinstance(val, list):
                summary[k] = val[:2]  # show first few answers/captions
            elif val is None:
                summary[k] = None
            else:
                summary[k] = f"<{type(val).__name__}>"
    pprint(summary)


if __name__ == "__main__":
    print_header("🖼️ COCO Captioning Dataset Test")
    try:
        coco_samples = load_coco_small(num_samples=2)
        print("✅ COCO loader executed successfully.")
        print("Sample COCO record:")
        summarize_record(coco_samples[0], keys=("image", "caption"))
    except Exception as e:
        print(f"❌ COCO loader failed: {e}")

    print_header("❓ VQAv2 Dataset Test")
    try:
        vqa_samples = load_vqa_small(num_samples=2)
        print("✅ VQAv2 loader executed successfully.")
        print("Sample VQA record:")
        summarize_record(vqa_samples[0], keys=("image", "question", "answers"))
    except Exception as e:
        print(f"❌ VQAv2 loader failed: {e}")

    print("\n✅ Dataset smoke test completed.\n")