"""
dataset_load_smoke.py
Runs sanity checks for COCO and VQAv2 dataset loaders.
"""
import os
import sys

# --- Fix Python path automatically ---
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from data_loaders.coco_loader import load_coco_small
from data_loaders.vqa_loader import load_vqa_small  # only if you use it


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
    print(summary)


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