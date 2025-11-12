"""
jsonl_conversion.py

Converts raw multimodal datasets (COCO, Flickr30k, VQAv2)
into standardized .jsonl files for downstream model training.

Each JSONL record is one sample:
- Caption datasets (COCO, Flickr30k): {"image_path": "...", "caption": "..."}
- VQA datasets (VQAv2): {"image_path": "...", "question": "...", "answer": "..."}

Environment configuration (.env):
    DATASETS_DIR=/path/to/all_datasets
    OUTPUT_JSONL_DIR=/path/to/output_preprocessed
"""

import os
import json
import csv
import time
from tqdm import tqdm
from dotenv import load_dotenv

# ----------- LOAD ENV VARIABLES -----------
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
OUTPUT_JSONL_DIR = os.getenv("OUTPUT_JSONL_DIR")

# ----------- VALIDATION CHECKS -----------
if not DATASETS_DIR or not OUTPUT_JSONL_DIR:
    raise ValueError(
        "❌ Missing environment variables. Please define DATASETS_DIR and OUTPUT_JSONL_DIR in your .env file."
    )

if not os.path.exists(DATASETS_DIR):
    raise FileNotFoundError(f"❌ DATASETS_DIR not found: {DATASETS_DIR}")

os.makedirs(OUTPUT_JSONL_DIR, exist_ok=True)

print(f"\n📁 DATASETS_DIR: {DATASETS_DIR}")
print(f"📂 OUTPUT_JSONL_DIR: {OUTPUT_JSONL_DIR}\n")

# ---------- TIMER UTILITY ----------
def timed_stage(stage_name, func, *args, **kwargs):
    print(f"\n🚀 Starting: {stage_name}")
    start_time = time.time()
    func(*args, **kwargs)
    elapsed = time.time() - start_time
    print(f"✅ Finished {stage_name} in {elapsed:.2f} sec\n")

# ---------- COCO --------------
def preprocess_coco(ann_path, split_name, output_path):
    print(f"🖼️  Processing COCO {split_name} ...")
    with open(ann_path, "r") as f:
        data = json.load(f)

    image_map = {img["id"]: img["file_name"] for img in data["images"]}
    annotations = data["annotations"]

    with open(output_path, "w") as out_f:
        for ann in tqdm(annotations, desc=f"COCO {split_name}", ncols=100):
            image_file = image_map.get(ann["image_id"])
            if image_file:
                entry = {
                    "image_path": f"COCO/{split_name}/{image_file}",
                    "caption": ann["caption"].strip()
                }
                out_f.write(json.dumps(entry) + "\n")

    print(f"📦 Saved: {output_path}")

# ---------- FLICKR30K --------------
def preprocess_flickr30k(csv_path, output_path):
    print("📸 Processing Flickr30k ...")

    with open(csv_path, newline="") as csvfile:
        # Read pipe-separated CSV
        reader = csv.DictReader(csvfile, delimiter='|')
        # Clean header names (strip whitespace)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = list(reader)

    seen = set()
    skipped = 0

    with open(output_path, "w") as out_f:
        for row in tqdm(rows, desc="Flickr30k", ncols=100):
            image_name = (row.get('image_name') or '').strip()
            caption_text = (row.get('comment') or '').strip()

            # Skip rows with missing data
            if not image_name or not caption_text:
                skipped += 1
                continue

            key = (image_name, caption_text)
            if key in seen:
                continue
            seen.add(key)

            entry = {
                "image_path": f"flickr30k/flickr30k_images/{image_name}",
                "caption": caption_text
            }
            out_f.write(json.dumps(entry) + "\n")

    print(f"📦 Saved: {output_path}")
    print(f"⚙️  Skipped {skipped} invalid or empty rows\n")

# ---------- VQAv2 --------------
def preprocess_vqa(question_path, annotation_path, split_name, output_path):
    print(f"💬 Processing VQAv2 {split_name} ...")
    with open(question_path, "r") as fq, open(annotation_path, "r") as fa:
        questions = json.load(fq)["questions"]
        annotations = json.load(fa)["annotations"]

    ann_map = {a["question_id"]: a for a in annotations}

    with open(output_path, "w") as out_f:
        for q in tqdm(questions, desc=f"VQAv2 {split_name}", ncols=100):
            qid = q["question_id"]
            image_id = q["image_id"]
            question = q["question"].strip()
            if qid not in ann_map:
                continue
            answer = ann_map[qid]["multiple_choice_answer"].strip()
            image_path = f"VQAv2/{split_name}/COCO_{split_name}_{image_id:012d}.jpg"
            entry = {
                "image_path": image_path,
                "question": question,
                "answer": answer
            }
            out_f.write(json.dumps(entry) + "\n")

    print(f"📦 Saved: {output_path}")

# ---------- COMBINE CAPTIONS --------------
def combine_captions(output_dir, files, combined_path):
    print("🔗 Combining caption datasets (COCO + Flickr30k)...")
    with open(combined_path, "w") as out_f:
        for file in tqdm(files, desc="Combining", ncols=100):
            with open(os.path.join(output_dir, file), "r") as f:
                for line in f:
                    out_f.write(line)
    print(f"📦 Combined captions saved: {combined_path}")

# ---------- MAIN RUNNER --------------
if __name__ == "__main__":
    print("🏗️  Starting Stage 0: JSONL Conversion\n")
    overall_start = time.time()

    # COCO
    timed_stage(
        "COCO Train",
        preprocess_coco,
        ann_path=f"{DATASETS_DIR}/COCO/annotations/captions_train2017.json",
        split_name="train2017",
        output_path=f"{OUTPUT_JSONL_DIR}/coco_train.jsonl"
    )
    timed_stage(
        "COCO Val",
        preprocess_coco,
        ann_path=f"{DATASETS_DIR}/COCO/annotations/captions_val2017.json",
        split_name="val2017",
        output_path=f"{OUTPUT_JSONL_DIR}/coco_val.jsonl"
    )

    # Flickr30k
    timed_stage(
        "Flickr30k",
        preprocess_flickr30k,
        csv_path=f"{DATASETS_DIR}/flickr30k/results.csv",
        output_path=f"{OUTPUT_JSONL_DIR}/flickr30k.jsonl"
    )

    # VQAv2
    timed_stage(
        "VQAv2 Train",
        preprocess_vqa,
        question_path=f"{DATASETS_DIR}/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json",
        annotation_path=f"{DATASETS_DIR}/VQAv2/v2_mscoco_train2014_annotations.json",
        split_name="train2014",
        output_path=f"{OUTPUT_JSONL_DIR}/vqa_train.jsonl"
    )
    timed_stage(
        "VQAv2 Val",
        preprocess_vqa,
        question_path=f"{DATASETS_DIR}/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json",
        annotation_path=f"{DATASETS_DIR}/VQAv2/v2_mscoco_val2014_annotations.json",
        split_name="val2014",
        output_path=f"{OUTPUT_JSONL_DIR}/vqa_val.jsonl"
    )

    # Combine COCO + Flickr30k
    timed_stage(
        "Combine Captions",
        combine_captions,
        OUTPUT_JSONL_DIR,
        ["coco_train.jsonl", "flickr30k.jsonl"],
        f"{OUTPUT_JSONL_DIR}/captions_combined.jsonl"
    )

    total_time = time.time() - overall_start
    print(f"🎉 Stage 0 complete in {total_time/60:.2f} minutes!")
    print(f"📁 JSONL files saved to: {OUTPUT_JSONL_DIR}\n")