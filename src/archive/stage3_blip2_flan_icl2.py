"""
Stage 3 — BLIP-2 (Flan-T5-XL) Few-Shot / In-Context Learning Captioning
-----------------------------------------------------------------------
Generates captions using BLIP-2 + Flan-T5-XL with configurable
few-shot examples (K) and prompt style (A/B/C).

Usage examples:
    python src/stage3_blip2_flan_icl.py --K 0 --prompt_style A   # Zero-shot baseline
    python src/stage3_blip2_flan_icl.py --K 3 --prompt_style A   # 3-shot explicit prompt
    python src/stage3_blip2_flan_icl.py --K 5 --prompt_style B   # 5-shot descriptive
    python src/stage3_blip2_flan_icl.py --K 3 --prompt_style C   # conversational
"""

import os, json, random, torch, argparse
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ------------------ CLI arguments ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=3, help="Number of few-shot examples (0 = zero-shot)")
parser.add_argument("--prompt_style", type=str, default="A", choices=["A", "B", "C"],
                    help="Prompt format: A=Explicit, B=Descriptive, C=Conversational")
args = parser.parse_args()

K = args.K
PROMPT_STYLE = args.prompt_style

# ------------------ Environment ------------------
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
OUTPUT_JSONL_DIR = os.getenv("OUTPUT_JSONL_DIR")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Salesforce/blip2-flan-t5-xl"

print(f"🚀 Using BLIP-2 model: {model_name}")
print(f"🧠 Mode: {K}-shot, Prompt style: {PROMPT_STYLE}")

# ------------------ Load model ------------------
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device).eval()
print(f"✅ Model loaded successfully on {device}")

# ------------------ Files ------------------
input_jsonl  = os.path.join(OUTPUT_JSONL_DIR, "captions_debug.jsonl")
output_jsonl = os.path.join(OUTPUT_JSONL_DIR, f"blip2_flan_icl_K{K}_{PROMPT_STYLE}.jsonl")

print(f"📂 Reading from: {input_jsonl}")
print(f"💾 Saving predictions to: {output_jsonl}")

# ------------------ Load dataset ------------------
all_samples = [json.loads(line) for line in open(input_jsonl)]
few_shot_pool = random.sample(all_samples, K) if K > 0 else []

# ------------------ Prompt builder ------------------
def build_prompt(k_examples, style="A"):
    """Return the text prompt for few-shot ICL."""
    if not k_examples:
        return "Describe the image in one complete sentence."

    if style == "A":
        examples_text = "\n".join(
            [f"Image {i+1}: {ex['caption']}" for i, ex in enumerate(k_examples)]
        )
        return f"{examples_text}\nNow describe the new image in one complete sentence."

    elif style == "B":
        examples_text = "\n".join(
            [f"Example {i+1}: {ex['caption']}" for i, ex in enumerate(k_examples)]
        )
        return (
            f"{examples_text}\nThe following image is new. "
            f"Provide a concise, factual caption describing its key objects and relationships."
        )

    elif style == "C":
        examples_text = "\n".join(
            [f"Q: What does the image show?\nA: {ex['caption']}" for ex in k_examples]
        )
        return f"{examples_text}\nQ: What does the new image show?\nA:"

prompt_template = build_prompt(few_shot_pool, PROMPT_STYLE)
print(f"\n🧩 Prompt preview:\n{'-'*40}\n{prompt_template}\n{'-'*40}\n")

# ------------------ Generation loop ------------------
count = 0
with open(output_jsonl, "w") as fout:
    for data in tqdm(all_samples, desc=f"Generating ({K}-shot, {PROMPT_STYLE})", ncols=100):
        img_path = os.path.join(DATASETS_DIR, data["image_path"])
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, text=prompt_template, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=40)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            data[f"generated_caption_K{K}_{PROMPT_STYLE}"] = caption
            fout.write(json.dumps(data) + "\n")
            count += 1

        except Exception as e:
            print(f"⚠️  Skipping {img_path} due to error: {e}")

print(f"\n✅ Finished! Generated captions for {count:,} samples.")
print(f"📁 Results saved at: {output_jsonl}")