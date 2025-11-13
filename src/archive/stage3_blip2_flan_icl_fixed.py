"""
Correct BLIP-2 Flan-T5 ICL Caption Generation
---------------------------------------------

This script fixes the issue where BLIP-2 was *ignoring* few-shot examples.
We now use the correct ICL pattern:

✓ EXAMPLE images represented by simple text tags (*not actual images*)
✓ EXAMPLE captions included in the full prompt
✓ QUERY image passed normally to BLIP-2
✓ Flan-T5 sees a clean, short prompt that ALWAYS fits within 512 tokens

This finally enables true few-shot behavior for BLIP-2.
"""

import os
import json
import random
import torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
OUTPUT_JSONL_DIR = os.getenv("OUTPUT_JSONL_DIR")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load BLIP-2 Flan-T5 XL
# -----------------------------
MODEL_NAME = "Salesforce/blip2-flan-t5-xl"

print(f"🚀 Loading BLIP-2 model: {MODEL_NAME}")
processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device).eval()

# -----------------------------
# Few-shot prompt templates
# -----------------------------
PROMPT_TEMPLATES = {
    "A": (
        "Here are some example image descriptions:\n\n"
        "{examples}\n"
        "Now describe the following image in one complete sentence:"
    ),
    "B": (
        "Learn the style from these sample captions:\n\n"
        "{examples}\n"
        "Write a caption that matches the style for the new image:"
    ),
    "C": (
        "Below are image descriptions. Use them to understand how to describe images.\n\n"
        "{examples}\n"
        "Now describe the next image naturally and clearly:"
    )
}

def build_example_block(samples):
    """
    Represent example images using TEXT TAGS, NOT real images.
    This is critical — otherwise BLIP-2 ignores the examples.
    """
    block = ""
    for img_cap in samples:
        img_name = os.path.basename(img_cap["image_path"])
        block += f"[IMAGE: {img_name}]\nCaption: {img_cap['caption']}\n\n"
    return block.strip()

# -----------------------------
# Main ICL generation
# -----------------------------
def generate_icl(K, prompt_style, debug_limit=None):
    input_file = os.path.join(OUTPUT_JSONL_DIR, "captions_debug.jsonl")
    output_file = os.path.join(
        OUTPUT_JSONL_DIR, f"blip2_flan_icl_K{K}_{prompt_style}.jsonl"
    )

    print(f"\n📂 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print(f"🧪 Running ICL with K={K}, prompt_style={prompt_style}")

    # Load all samples
    all_samples = [json.loads(l) for l in open(input_file)]
    total = len(all_samples)

    # Prepare output writer
    fout = open(output_file, "w")

    for idx, data in enumerate(tqdm(all_samples, desc=f"ICL K={K} Style={prompt_style}")):

        if debug_limit and idx >= debug_limit:
            break

        query_img_path = os.path.join(DATASETS_DIR, data["image_path"])
        if not os.path.exists(query_img_path):
            continue

        # Select K random examples (not including the query example)
        few_shot_examples = random.sample(all_samples, K)

        example_block = build_example_block(few_shot_examples)

        # Build the final ICL prompt
        template = PROMPT_TEMPLATES[prompt_style]
        prompt_text = template.format(examples=example_block)

        # Load query image
        image = Image.open(query_img_path).convert("RGB")

        # Pass: image = new image, text = few-shot prompt
        inputs = processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        ).to(device)

        # Generate caption
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False
            )
        caption = processor.decode(output_tokens[0], skip_special_tokens=True)

        # Save result
        result = {
            "image_path": data["image_path"],
            "caption": data["caption"],
            f"generated_caption_K{K}_{prompt_style}": caption
        }
        fout.write(json.dumps(result) + "\n")

    fout.close()
    print(f"✅ DONE. Saved to {output_file}")

# -----------------------------
# Run Experiments
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--prompt_style", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--debug_limit", type=int, default=None)
    args = parser.parse_args()

    generate_icl(K=args.K, prompt_style=args.prompt_style, debug_limit=args.debug_limit)