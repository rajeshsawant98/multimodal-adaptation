"""
BLIP-2 OPT-2.7B — Final ICL Captioning Script (No Prefix)
--------------------------------------------------------
Outputs are now saved under:
    $EXPERIMENTS_DIR/opt/
"""

import os, json, random, torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -----------------------------
# Environment paths
# -----------------------------
load_dotenv()

DATASETS_DIR = os.getenv("DATASETS_DIR")
PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR")            # captions_debug.jsonl lives here
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", PREPROCESSED_DIR)

# create subfolder for OPT experiments
OPT_DIR = os.path.join(EXPERIMENTS_DIR, "opt")
os.makedirs(OPT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Model
# -----------------------------
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
print(f"\n🚀 Loading model: {MODEL_NAME}")

processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device).eval()

# -----------------------------
# PROMPT TEMPLATES
# (No prefix, only examples)
# -----------------------------
PROMPT_TEMPLATES = {
    "A": "{examples}",
    "B": "{examples}",
    "C": "{examples}",
}

# -----------------------------
# Build few-shot examples
# -----------------------------
def build_example_block(samples, max_words=10):
    """
    Minimal format that works with OPT:
      Example 1: a man riding a bike.
      Example 2: a blue bathroom sink.
    """
    if not samples:
        return ""

    lines = []
    for i, s in enumerate(samples, start=1):
        cap = s["caption"]
        cap = " ".join(cap.split()[:max_words]).strip()
        lines.append(f"Example {i}: {cap}.")
    return "\n".join(lines) + "\n\n"


# -----------------------------
# Main generator
# -----------------------------
def generate_opt_icl(K, prompt_style, debug_limit=None):

    input_file = os.path.join(PREPROCESSED_DIR, "captions_debug.jsonl")

    # NOW OUTPUTS GO INTO: $EXPERIMENTS_DIR/opt/
    output_file = os.path.join(
        OPT_DIR,
        f"blip2_opt_icl_K{K}_{prompt_style}.jsonl"
    )

    print(f"\n📂 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print(f"🔧 OPT ICL — K={K}, style={prompt_style}")

    all_samples = [json.loads(l) for l in open(input_file)]
    fout = open(output_file, "w")

    for idx, sample in enumerate(
        tqdm(all_samples, desc=f"OPT-ICL K={K} style={prompt_style}")
    ):
        if debug_limit and idx >= debug_limit:
            break

        img_path = os.path.join(DATASETS_DIR, sample["image_path"])
        if not os.path.exists(img_path):
            continue

        # ----- Few-shot sampling -----
        few_shots = random.sample(all_samples, K) if K > 0 else []
        example_block = build_example_block(few_shots)

        template = PROMPT_TEMPLATES[prompt_style]
        prompt_text = template.format(examples=example_block)

        image = Image.open(img_path).convert("RGB")

        inputs = processor(
            images=image,
            text=prompt_text,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # ----- Generate -----
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=25,
                do_sample=False,
                num_beams=3,
            )
        caption = processor.decode(output_tokens[0], skip_special_tokens=True).strip()

        # ----- Write -----
        fout.write(json.dumps({
            "image_path": sample["image_path"],
            "caption": sample["caption"],
            f"generated_caption_K{K}_{prompt_style}": caption
        }) + "\n")

    fout.close()
    print(f"✅ DONE — saved to: {output_file}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--prompt_style", type=str, choices=["A","B","C"], default="A")
    parser.add_argument("--debug_limit", type=int, default=None)

    args = parser.parse_args()
    generate_opt_icl(args.K, args.prompt_style, args.debug_limit)