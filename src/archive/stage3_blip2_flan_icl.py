"""
BLIP-2 Flan-T5 XL — In-Context Captioning (ICL)
----------------------------------------------
- Model: Salesforce/blip2-flan-t5-xl
- Typical use in our project: K = 0 (zero-shot baseline)
- Prompt styles: A / B / C
- Short, hallucination-aware prompts
- Debug limit for quick runs
- Writes outputs under: $EXPERIMENTS_DIR/flan/
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
# Environment
# -----------------------------
load_dotenv()

DATASETS_DIR = os.getenv("DATASETS_DIR")
PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR")        # where captions_debug.jsonl lives
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", PREPROCESSED_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

if DATASETS_DIR is None or PREPROCESSED_DIR is None:
    raise ValueError("DATASETS_DIR and OUTPUT_JSONL_DIR must be set in .env")

# -----------------------------
# Model (Flan-only)
# -----------------------------
MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
print(f"\n🚀 Loading BLIP-2 Flan model: {MODEL_NAME}")

processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device).eval()

# -----------------------------
# Prompt templates (short + safe)
# -----------------------------
PROMPT_TEMPLATES = {
    # Factual / safe
    "A": (
        "You write short, factual captions for images.\n"
        "Describe only what is clearly visible.\n"
        "Do not guess hidden details.\n\n"
        "{examples}"
        "Caption:"
    ),
    # Descriptive but grounded
    "B": (
        "Here are example image captions.\n"
        "Follow their style: concise and accurate.\n\n"
        "{examples}"
        "Caption:"
    ),
    # Conversational but grounded
    "C": (
        "Look at the examples and learn the style.\n"
        "Now describe the new image naturally but briefly.\n\n"
        "{examples}"
        "Caption:"
    ),
}

# -----------------------------
# Utilities
# -----------------------------
def build_example_block(samples, max_words: int = 12) -> str:
    """
    Build a compact example block.

    Format:
      Example 1: a man riding a bike.
      Example 2: a blue bathroom with wall sink.
    """
    if not samples:
        return ""  # for K = 0

    lines = []
    for i, s in enumerate(samples, start=1):
        cap = s["caption"]
        cap = " ".join(cap.split()[:max_words]).strip()
        lines.append(f"Example {i}: {cap}")
    return "\n".join(lines) + "\n\n"


def cleanup_caption(text: str) -> str:
    """Strip boilerplate, keep only the first sentence."""
    text = text.strip().strip('"')

    lower = text.lower()
    if lower.startswith("caption:"):
        text = text[len("caption:"):].strip()

    # keep only first sentence if there are multiple
    if "." in text:
        text = text.split(".")[0].strip()

    return text


# -----------------------------
# Main ICL generator
# -----------------------------
def generate_icl(K: int, prompt_style: str, debug_limit: int | None = None):
    input_file = os.path.join(PREPROCESSED_DIR, "captions_debug.jsonl")

    flan_dir = os.path.join(EXPERIMENTS_DIR, "flan")
    os.makedirs(flan_dir, exist_ok=True)

    output_file = os.path.join(
        flan_dir,
        f"blip2_flan_icl_K{K}_{prompt_style}.jsonl",
    )

    print(f"\n📂 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print(f"🔧 Flan ICL — K={K}, style={prompt_style}")

    with open(input_file, "r") as f:
        all_samples = [json.loads(line) for line in f]

    with open(output_file, "w") as fout:
        for idx, sample in enumerate(
            tqdm(all_samples, desc=f"Flan K={K} style={prompt_style}")
        ):
            if debug_limit is not None and idx >= debug_limit:
                break

            img_path = os.path.join(DATASETS_DIR, sample["image_path"])
            if not os.path.exists(img_path):
                continue

            # ---- choose few-shot examples (for K > 0) ----
            if K > 0:
                few_shots = random.sample(all_samples, K)
            else:
                few_shots = []

            example_block = build_example_block(few_shots)
            template = PROMPT_TEMPLATES[prompt_style]
            prompt_text = template.format(examples=example_block)

            # ---- load query image ----
            image = Image.open(img_path).convert("RGB")

            # ---- encode image + prompt ----
            inputs = processor(
                images=image,
                text=prompt_text,
                return_tensors="pt",
            ).to(device)

            # ---- generate caption ----
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,    # deterministic
                    num_beams=3,        # small beam search for quality
                    length_penalty=0.9,
                )

            raw_caption = processor.decode(
                output_tokens[0], skip_special_tokens=True
            )
            caption = cleanup_caption(raw_caption)

            fout.write(json.dumps({
                "image_path": sample["image_path"],
                "caption": sample["caption"],
                f"generated_caption_K{K}_{prompt_style}": caption,
            }) + "\n")

    print(f"✅ Done. Saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=0, help="number of ICL examples")
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="A",
        choices=["A", "B", "C"],
        help="prompt template variant",
    )
    parser.add_argument(
        "--debug_limit",
        type=int,
        default=None,
        help="limit number of samples (for quick tests)",
    )

    args = parser.parse_args()
    generate_icl(K=args.K, prompt_style=args.prompt_style, debug_limit=args.debug_limit)