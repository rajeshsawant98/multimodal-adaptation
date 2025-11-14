"""
InstructBLIP – Short, Factual Captioning with ICL
------------------------------------------------
No 'clean' in naming. Produces short, factual captions.
"""

import os
import json
import random
import torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()
DATASETS_DIR     = os.getenv("DATASETS_DIR")
PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR")
EXPERIMENTS_DIR  = os.getenv("EXPERIMENTS_DIR", PREPROCESSED_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load InstructBLIP
# --------------------------------------------------
MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"
print(f"\n🚀 Loading InstructBLIP: {MODEL_NAME}")

processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device).eval()

# --------------------------------------------------
# Prompt templates (short & factual)
# --------------------------------------------------
PROMPT_TEMPLATES = {
    "A": (
        "Write a short, factual caption.\n"
        "{examples}"
        "Caption:"
    ),
    "B": (
        "Follow the style of the examples: brief and factual.\n\n"
        "{examples}"
        "Caption:"
    ),
    "C": (
        "Learn from the examples and describe the new image briefly.\n\n"
        "{examples}"
        "Caption:"
    ),
}

def build_example_block(samples, max_words=10):
    """Compact example structure for ICL."""
    if not samples:
        return ""
    lines = []
    for i, s in enumerate(samples, start=1):
        cap = " ".join(s["caption"].split()[:max_words])
        lines.append(f"Example {i}: {cap}.")
    return "\n".join(lines) + "\n\n"

# --------------------------------------------------
# Caption generation
# --------------------------------------------------
def generate_captions(K, prompt_style, debug_limit=None):
    input_file = os.path.join(PREPROCESSED_DIR, "captions_debug.jsonl")

    # updated: no 'clean' in folder name
    output_dir = os.path.join(EXPERIMENTS_DIR, "instructblip")
    os.makedirs(output_dir, exist_ok=True)

    # updated: no 'clean' in filename
    output_file = os.path.join(
        output_dir, f"instructblip_K{K}_{prompt_style}.jsonl"
    )

    print(f"\n📂 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print(f"🔧 InstructBLIP Captioning — K={K}, style={prompt_style}")

    all_samples = [json.loads(x) for x in open(input_file)]
    fout = open(output_file, "w")

    for idx, sample in enumerate(
        tqdm(all_samples, desc=f"INSTRUCTBLIP K={K} style={prompt_style}")
    ):
        if debug_limit is not None and idx >= debug_limit:
            break

        img_path = os.path.join(DATASETS_DIR, sample["image_path"])
        if not os.path.exists(img_path):
            continue

        few_shots = random.sample(all_samples, K) if K > 0 else []
        example_block = build_example_block(few_shots)

        prompt = PROMPT_TEMPLATES[prompt_style].format(examples=example_block)

        image = Image.open(img_path).convert("RGB")

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                num_beams=4,
                length_penalty=0.9,
            )

        caption = processor.decode(tokens[0], skip_special_tokens=True).strip()

        fout.write(json.dumps({
            "image_path": sample["image_path"],
            "caption": sample["caption"],
            f"generated_caption_K{K}_{prompt_style}": caption,
        }) + "\n")

    fout.close()
    print(f"✅ DONE → {output_file}")

# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--prompt_style", type=str, choices=["A","B","C"], default="A")
    parser.add_argument("--debug_limit", type=int, default=None)

    args = parser.parse_args()
    generate_captions(args.K, args.prompt_style, args.debug_limit)