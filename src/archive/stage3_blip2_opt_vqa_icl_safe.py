"""
BLIP-2 OPT VQA ICL — Clean (No Normalization)
----------------------------------------------
Features:
✓ Clean raw extraction after “Answer:”
✓ No normalization at all (exact text preserved)
✓ Stable ICL prompt for OPT
"""

import os
import json
import random
import re
import torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from collections import Counter
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ============================================================
# Environment
# ============================================================
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR")
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", PREPROCESSED_DIR)

OUT_DIR = os.path.join(EXPERIMENTS_DIR, "vqa_opt_clean")
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Load BLIP-2 OPT
# ============================================================
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
print(f"\n🚀 Loading BLIP-2 OPT model: {MODEL_NAME}")

processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device).eval()


# ============================================================
# Extract only answer (no normalization)
# ============================================================
def extract_after_answer(raw: str) -> str:
    """Extract only the answer text after 'Answer:'."""
    if "Answer:" not in raw:
        return raw.strip()

    text = raw.split("Answer:", 1)[1].strip()

    # Stop at newline
    text = text.split("\n")[0].strip()

    # Clean trailing punctuation
    text = text.rstrip(" .,")

    # Clip to ~5–6 words for stability
    words = text.split()
    if len(words) > 6:
        words = words[:6]
    return " ".join(words)


# ============================================================
# Build Few-Shot Examples
# ============================================================
def get_gt(sample):
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

def build_examples(samples, K):
    if K == 0:
        return ""
    shots = random.sample(samples, K)

    lines = []
    for s in shots:
        q = s["question"]
        a = get_gt(s)
        lines.append(f"Q: {q}\nA: {a}")
    return "\n".join(lines) + "\n\n"


# ============================================================
# Main VQA ICL Loop
# ============================================================
def run_vqa_icl(K=0, debug_limit=None):
    input_file = os.path.join(PREPROCESSED_DIR, "vqa_debug.jsonl")
    out_file = os.path.join(OUT_DIR, f"vqa_opt_K{K}.jsonl")

    print(f"\n📂 Input:  {input_file}")
    print(f"💾 Output: {out_file}")
    print(f"🔧 BLIP-2 OPT VQA — K={K} (NO NORMALIZATION)")

    samples = [json.loads(l) for l in open(input_file)]
    fout = open(out_file, "w")

    for idx, sample in enumerate(tqdm(samples, desc=f"VQA K={K}")):
        if debug_limit and idx >= debug_limit:
            break

        gt = get_gt(sample)

        # Image
        img_path = os.path.join(DATASETS_DIR, sample["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        # Prompt
        examples_block = build_examples(samples, K)
        prompt = f"{examples_block}Question: {sample['question']}\nAnswer:"

        # Encode
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=15,
                num_beams=3,
                do_sample=False,
            )

        raw = processor.decode(out[0], skip_special_tokens=True)
        short = extract_after_answer(raw)

        # Save
        fout.write(json.dumps({
            "image_path": sample["image_path"],
            "question": sample["question"],
            "answer_gt": gt,
            "answer_raw": raw,
            f"answer_K{K}": short
        }) + "\n")

    fout.close()
    print(f"✅ DONE → {out_file}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--debug_limit", type=int, default=None)
    args = parser.parse_args()
    run_vqa_icl(K=args.K, debug_limit=args.debug_limit)