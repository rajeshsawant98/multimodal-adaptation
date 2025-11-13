"""
BLIP-2 OPT — Simple Instruction ICL Captioning
---------------------------------------------
Stable for OPT-2.7B
✓ Does NOT repeat examples
✓ Follows instruction
✓ Good for K=0,1,3,5
"""

import os, json, random, torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -----------------------------
# Environment
# -----------------------------
load_dotenv()
DATASETS_DIR      = os.getenv("DATASETS_DIR")
PREPROCESSED_DIR  = os.getenv("OUTPUT_JSONL_DIR")
EXPERIMENTS_DIR   = os.getenv("EXPERIMENTS_DIR", PREPROCESSED_DIR)

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load BLIP-2 OPT
# -----------------------------
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
print(f"\n🚀 Loading OPT: {MODEL_NAME}")

processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device).eval()

# -----------------------------
# Build short examples
# -----------------------------
def build_examples(samples, max_words=12):
    """
    Produces:
    Example 1: a man riding a bike.
    Example 2: a blue bathroom with wall sink.
    """
    if not samples:
        return ""

    lines = []
    for i, s in enumerate(samples, start=1):
        cap = " ".join(s["caption"].split()[:max_words]).strip()
        lines.append(f"Example {i}: {cap}.")
    return "\n".join(lines) + "\n\n"


# -----------------------------
# Main ICL generation
# -----------------------------
def run_opt_icl(K, debug_limit=None):
    input_file  = os.path.join(PREPROCESSED_DIR, "captions_debug.jsonl")
    output_file = os.path.join(EXPERIMENTS_DIR, f"opt_K{K}.jsonl")

    print(f"\n📂 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print(f"🔧 Running OPT ICL, K={K}")

    all_samples = [json.loads(l) for l in open(input_file)]
    fout = open(output_file, "w")

    for idx, row in enumerate(tqdm(all_samples, desc=f"OPT K={K}")):
        if debug_limit and idx >= debug_limit:
            break

        img_path = os.path.join(DATASETS_DIR, row["image_path"])
        if not os.path.exists(img_path):
            continue

        # Select K examples
        few_shots = random.sample(all_samples, K) if K > 0 else []
        example_text = build_examples(few_shots)

        # ----- FINAL SIMPLE INSTRUCTION PROMPT -----
        prompt = (
            "Instruction: Describe the image in one short, factual sentence.\n\n"
            f"{example_text}"
            "Now describe the new image:\n"
        )

        # Load query image
        image = Image.open(img_path).convert("RGB")

        # Encode
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device)

        # Generate
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                num_beams=3,
                repetition_penalty=1.0,
            )

        caption = processor.decode(out[0], skip_special_tokens=True).strip()

        fout.write(json.dumps({
            "image_path": row["image_path"],
            "caption": row["caption"],
            f"generated_caption_K{K}": caption
        }) + "\n")

    fout.close()
    print(f"✅ DONE → {output_file}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--K", type=int, default=0)
    p.add_argument("--debug_limit", type=int, default=None)

    args = p.parse_args()
    run_opt_icl(args.K, args.debug_limit)