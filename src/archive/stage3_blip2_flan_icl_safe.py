"""
BLIP-2 FLAN-T5 XL — Safer ICL Captioning
Better language style, but more hallucination-prone.

Safety additions:
✔ grounding rules
✔ “visible only” instruction
✔ capped example length
✔ deterministic decoding
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
DATASETS_DIR = os.getenv("DATASETS_DIR")
OUTPUT_JSONL_DIR = os.getenv("OUTPUT_JSONL_DIR")
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Model
# -----------------------------
MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
print(f"\n🚀 Loading BLIP-2 FLAN model: {MODEL_NAME}")

processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device).eval()

# -----------------------------
# Prompt Templates
# -----------------------------
PROMPT_TEMPLATES = {
    "A": (
        "Below are example captions. They are short, factual, and describe only visible content.\n"
        "Follow their style.\n\n"
        "{examples}\n"
        "Now accurately describe the new image:"
    ),
    "B": (
        "Learn the captioning pattern from the examples.\n"
        "Avoid imagining objects that are not visible.\n\n"
        "{examples}\n"
        "Caption the new image concisely:"
    ),
    "C": (
        "These are sample grounded image descriptions.\n"
        "Use them to guide your new caption.\n\n"
        "{examples}\n"
        "Now describe the next image:"
    )
}

def build_example_block(samples):
    block = ""
    for s in samples:
        img = os.path.basename(s["image_path"])
        cap = " ".join(s["caption"].split()[:10])
        block += f"[IMG: {img}]\nCaption: {cap}\n\n"
    return block.strip()

# -----------------------------
# ICL Generation Loop
# -----------------------------
def generate_icl(K, prompt_style, debug_limit=None):

    input_file = os.path.join(OUTPUT_JSONL_DIR, "captions_debug.jsonl")
    output_file = os.path.join(OUTPUT_JSONL_DIR, f"blip2_flan_K{K}_{prompt_style}.jsonl")

    print(f"\n📂 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print(f"🔧 Mode: FLAN, K={K}, style={prompt_style}")

    all_samples = [json.loads(x) for x in open(input_file)]
    fout = open(output_file, "w")

    for idx, sample in enumerate(tqdm(all_samples, desc=f"FLAN K={K} style={prompt_style}")):

        if debug_limit and idx >= debug_limit:
            break

        img_path = os.path.join(DATASETS_DIR, sample["image_path"])
        if not os.path.exists(img_path):
            continue

        few_shots = random.sample(all_samples, K)
        example_block = build_example_block(few_shots)

        prompt = PROMPT_TEMPLATES[prompt_style].format(examples=example_block)

        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0
            )

        caption = processor.decode(tokens[0], skip_special_tokens=True)

        fout.write(json.dumps({
            "image_path": sample["image_path"],
            "caption": sample["caption"],
            f"generated_caption_K{K}_{prompt_style}": caption
        }) + "\n")

    fout.close()
    print(f"✅ Completed → {output_file}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--prompt_style", type=str, choices=["A","B","C"], default="A")
    p.add_argument("--debug_limit", type=int, default=None)
    args = p.parse_args()

    generate_icl(args.K, args.prompt_style, args.debug_limit)