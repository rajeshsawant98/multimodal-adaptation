"""
BLIP-2 OPT VQA ICL Script (Stable Version)
-------------------------------------------
- Supports K-shot few-shot in-context learning
- Automatically extracts ground-truth (answer_gt)
- Works with any standard VQA v2 JSON structure
- Clean prompts to reduce hallucination
"""

import os
import json
import random
import torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from collections import Counter
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -----------------------------
# Environment Setup
# -----------------------------
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR")
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", PREPROCESSED_DIR)
OUT_DIR = os.path.join(EXPERIMENTS_DIR, "vqa_opt")
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load BLIP-2 OPT
# -----------------------------
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
print(f"\n🚀 Loading VQA model: {MODEL_NAME}")

processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device).eval()


# -----------------------------
# Clean VQA Prompt Template
# -----------------------------
def make_prompt(question, examples_block):
    """
    Final prompt sent to OPT. Keep it extremely simple.
    """
    if examples_block:
        return f"{examples_block}Question: {question}\nAnswer:"
    else:
        return f"Question: {question}\nAnswer:"


# -----------------------------
# Few-shot Example Builder
# -----------------------------
def build_examples(samples, K=0):
    if K == 0:
        return ""

    shots = random.sample(samples, K)

    lines = []
    for s in shots:
        q = s["question"]

        # Extract ground truth safely
        if "answer_gt" in s:
            a = s["answer_gt"]
        elif "answer" in s:
            a = s["answer"]
        elif "multiple_choice_answer" in s:
            a = s["multiple_choice_answer"]
        elif "answers" in s and isinstance(s["answers"], list):
            cnt = Counter([x["answer"] for x in s["answers"]])
            a = cnt.most_common(1)[0][0]
        else:
            a = ""

        lines.append(f"Q: {q}\nA: {a}")

    return "\n".join(lines) + "\n\n"


# -----------------------------
# Main VQA ICL Loop
# -----------------------------
def run_vqa_icl(K=0, debug_limit=None):

    input_file = os.path.join(PREPROCESSED_DIR, "vqa_debug.jsonl")
    output_file = os.path.join(OUT_DIR, f"vqa_opt_K{K}.jsonl")

    print(f"\n📂 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print(f"🔧 Mode: OPT VQA — K={K}")

    all_samples = [json.loads(l) for l in open(input_file)]

    fout = open(output_file, "w")

    for idx, sample in enumerate(
        tqdm(all_samples, desc=f"VQA K={K}")
    ):
        if debug_limit and idx >= debug_limit:
            break

        # ------------------------------
        # Extract safe ground truth
        # ------------------------------
        if "answer_gt" in sample:
            gt = sample["answer_gt"]
        elif "answer" in sample:
            gt = sample["answer"]
        elif "multiple_choice_answer" in sample:
            gt = sample["multiple_choice_answer"]
        elif "answers" in sample and isinstance(sample["answers"], list):
            cnt = Counter([x["answer"] for x in sample["answers"]])
            gt = cnt.most_common(1)[0][0]
        else:
            gt = ""

        # ------------------------------
        # Prepare image
        # ------------------------------
        img_path = os.path.join(DATASETS_DIR, sample["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue

        # ------------------------------
        # Build K-shot prompt
        # ------------------------------
        examples_block = build_examples(all_samples, K)
        prompt = make_prompt(sample["question"], examples_block)

        # ------------------------------
        # BLIP-2 encode
        # ------------------------------
        inputs = processor(
            images=image,
            text=prompt,
            padding=True,
            return_tensors="pt"
        ).to(device)

        # ------------------------------
        # Generate answer
        # ------------------------------
        with torch.no_grad():
            out_tokens = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                num_beams=3,
            )

        pred = processor.decode(out_tokens[0], skip_special_tokens=True).strip()

        # ------------------------------
        # Save result
        # ------------------------------
        fout.write(json.dumps({
            "image_path": sample["image_path"],
            "question": sample["question"],
            "answer_gt": gt,
            f"answer_K{K}": pred
        }) + "\n")

    fout.close()
    print(f"✅ DONE: {output_file}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--debug_limit", type=int, default=None)

    args = parser.parse_args()
    run_vqa_icl(K=args.K, debug_limit=args.debug_limit)