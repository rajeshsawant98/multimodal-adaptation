"""
Run VQA ICL over the FULL VQAv2 TRAIN SET
-----------------------------------------
- Streams through vqa_train.jsonl (no RAM explosion)
- Supports K-shot prompting
- Uses type-aware few-shot examples
- Stores output in shards (50k per file)
"""

import os, json, random, torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from collections import Counter
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ---------------------------------------------------
# ENV
# ---------------------------------------------------
load_dotenv()
DATASETS_DIR      = os.getenv("DATASETS_DIR")
PREPROCESSED_DIR  = os.getenv("OUTPUT_JSONL_DIR")
EXPERIMENTS_DIR   = os.getenv("EXPERIMENTS_DIR", PREPROCESSED_DIR)
OUT_DIR           = os.path.join(EXPERIMENTS_DIR, "vqa_opt_full")
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
print(f"\n🚀 Loading BLIP-2 OPT: {MODEL_NAME}")

processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device).eval()

# ---------------------------------------------------
# Utility: safe GT extraction
# ---------------------------------------------------
def extract_gt(sample):
    if "answer_gt" in sample:
        return sample["answer_gt"]
    if "answer" in sample:
        return sample["answer"]
    if "multiple_choice_answer" in sample:
        return sample["multiple_choice_answer"]
    if "answers" in sample:
        cnt = Counter([a["answer"] for a in sample["answers"]])
        return cnt.most_common(1)[0][0]
    return ""

# ---------------------------------------------------
# TYPE MAP (from your analysis script)
# ---------------------------------------------------
def detect_qtype(q):
    q_lower = q.lower()
    if q_lower.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "should ")):
        return "yes/no"
    if q_lower.startswith(("how many", "how much")):
        return "count"
    if "color" in q_lower:
        return "color"
    if q_lower.startswith("where"):
        return "location"
    if q_lower.startswith("what type") or "kind" in q_lower:
        return "type"
    return "other"

# ---------------------------------------------------
# Load all training samples ONCE for few-shot use
# ---------------------------------------------------
print("\n📥 Loading full VQA train dataset into memory for ICL sampling...")
TRAIN_FILE = os.path.join(PREPROCESSED_DIR, "vqa_train.jsonl")

all_train = []
with open(TRAIN_FILE) as f:
    for line in f:
        row = json.loads(line)
        row["qtype"] = detect_qtype(row["question"])
        all_train.append(row)

print(f"✔ Loaded {len(all_train):,} train samples.")

# Index by type
by_type = {}
for row in all_train:
    t = row["qtype"]
    by_type.setdefault(t, []).append(row)

# ---------------------------------------------------
# Few-shot builder
# ---------------------------------------------------
def build_examples(qtype, K):
    if K == 0:
        return ""

    pool = by_type.get(qtype, [])
    if len(pool) < K:
        pool = all_train

    shots = random.sample(pool, K)

    lines = []
    for s in shots:
        a = extract_gt(s)
        lines.append(f"Q: {s['question']}\nA: {a}")

    return "\n".join(lines) + "\n\n"

# ---------------------------------------------------
# Main full-train runner
# ---------------------------------------------------
def run_full_train(K=0, shard_size=50000):

    out_shard_idx = 0
    out_count = 0
    fout = None

    def open_new_shard():
        nonlocal fout, out_shard_idx
        if fout:
            fout.close()
        shard_path = os.path.join(OUT_DIR, f"vqa_opt_full_K{K}_shard{out_shard_idx}.jsonl")
        fout = open(shard_path, "w")
        print(f"📝 Opened shard {shard_path}")
        out_shard_idx += 1

    open_new_shard()

    # Stream through full dataset line-by-line
    with open(TRAIN_FILE) as f:
        for raw_idx, line in enumerate(tqdm(f, desc=f"FULL VQA K={K}")):

            row = json.loads(line)

            img_path = os.path.join(DATASETS_DIR, row["image_path"])
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue

            q = row["question"]
            qtype = detect_qtype(q)
            gt = extract_gt(row)

            examples = build_examples(qtype, K)
            prompt = f"{examples}Question: {q}\nAnswer:"

            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                out_tokens = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    num_beams=3,
                    do_sample=False
                )

            answer_raw = processor.decode(out_tokens[0], skip_special_tokens=True).strip()

            # Remove prompt prefix
            if "Answer:" in answer_raw:
                pred = answer_raw.split("Answer:", 1)[1].strip()
            else:
                pred = answer_raw.strip()

            # Write record
            fout.write(json.dumps({
                "image_path": row["image_path"],
                "question": q,
                "qtype": qtype,
                "answer_gt": gt,
                "answer_raw": answer_raw,
                f"answer_K{K}": pred
            }) + "\n")

            out_count += 1

            # Rotate shard
            if out_count % shard_size == 0:
                open_new_shard()

    if fout:
        fout.close()
    print("\n🎉 DONE — full dataset processed.")

# ---------------------------------------------------
# CLI
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=50000)
    args = parser.parse_args()

    run_full_train(args.K, args.shard_size)