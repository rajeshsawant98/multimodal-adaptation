"""
validate_jsonl_datasets.py

Quick sanity check for all generated JSONL datasets:
- Confirms file existence and line counts
- Checks sample structure (captioning vs. VQA)
- Verifies image paths exist
- Prints stats on missing or invalid entries
"""

import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

# ---------- LOAD ENVIRONMENT ----------
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
OUTPUT_JSONL_DIR = os.getenv("OUTPUT_JSONL_DIR")

if not DATASETS_DIR or not OUTPUT_JSONL_DIR:
    raise ValueError("❌ Missing environment variables in .env (DATASETS_DIR, OUTPUT_JSONL_DIR).")

if not os.path.exists(DATASETS_DIR):
    raise FileNotFoundError(f"❌ DATASETS_DIR not found: {DATASETS_DIR}")

if not os.path.exists(OUTPUT_JSONL_DIR):
    raise FileNotFoundError(f"❌ OUTPUT_JSONL_DIR not found: {OUTPUT_JSONL_DIR}")


# ---------- VALIDATION FUNCTION ----------
def validate_jsonl(file_path, is_vqa=False, sample_limit=5):
    print(f"\n🔍 Validating: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    total = 0
    missing_fields = 0
    missing_images = 0
    first_samples = []

    # Determine schema for validation
    expected = {"image_path", "question", "answer"} if is_vqa else {"image_path", "caption"}

    with open(file_path, "r") as f:
        for i, line in enumerate(tqdm(f, desc=os.path.basename(file_path), ncols=100)):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"❌ JSON parse error at line {i + 1}")
                continue

            # Verify required fields exist
            if not expected.issubset(data.keys()):
                missing_fields += 1
                continue

            # Check if image path exists relative to DATASETS_DIR
            image_abs = os.path.join(DATASETS_DIR, data["image_path"])
            if not os.path.exists(image_abs):
                missing_images += 1

            # Collect a few example samples for inspection
            if len(first_samples) < sample_limit:
                first_samples.append(data)

            total += 1

    # ---------- SUMMARY ----------
    print(f"📊 Total samples: {total:,}")
    print(f"⚠️  Missing fields: {missing_fields}")
    print(f"🖼️  Missing image files: {missing_images}")
    valid = total - missing_fields - missing_images
    print(f"✅ Structure OK for {valid:,} entries")

    print("\n🔹 Example samples:")
    for s in first_samples:
        print(json.dumps(s, indent=2))
    print("-" * 60)


# ---------- MAIN ----------
if __name__ == "__main__":
    print(f"\n📂 Validating datasets in: {OUTPUT_JSONL_DIR}\n")

    files = {
        "coco_train.jsonl": False,
        "coco_val.jsonl": False,
        "flickr30k.jsonl": False,
        "captions_combined.jsonl": False,
        "vqa_train.jsonl": True,
        "vqa_val.jsonl": True,
    }

    for fname, is_vqa in files.items():
        path = os.path.join(OUTPUT_JSONL_DIR, fname)
        if os.path.exists(path):
            validate_jsonl(path, is_vqa=is_vqa)
        else:
            print(f"⚠️  Skipping (not found): {fname}")

    print("\n✅ Validation complete!\n")