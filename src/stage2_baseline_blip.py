"""
Runs a baseline caption generation using the BLIP model.
Input: captions_combined.jsonl
Output: baseline_blip_preds.jsonl
"""

import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv

# -------- LOAD ENV -----------
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
OUTPUT_JSONL_DIR = os.getenv("OUTPUT_JSONL_DIR")

# -------- MODEL INIT -----------
print("🚀 Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu").eval()
device = next(model.parameters()).device
print(f"✅ Model loaded on {device}")

# -------- INPUT/OUTPUT PATHS -----------

#input_jsonl = os.path.join(OUTPUT_JSONL_DIR, "captions_combined.jsonl")
input_jsonl = os.path.join(OUTPUT_JSONL_DIR, "captions_debug.jsonl")
output_jsonl = os.path.join(OUTPUT_JSONL_DIR, "baseline_blip_preds.jsonl")

# -------- PROCESS IMAGES -----------
print(f"📂 Reading from: {input_jsonl}")
print(f"💾 Saving predictions to: {output_jsonl}")

count = 0
with open(input_jsonl, "r") as fin, open(output_jsonl, "w") as fout:
    for line in tqdm(fin, desc="Generating captions", ncols=100):
        data = json.loads(line)
        image_path = os.path.join(DATASETS_DIR, data["image_path"])

        # Skip if image missing
        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, max_new_tokens=30)
            generated_caption = processor.decode(output_ids[0], skip_special_tokens=True)

            # Write out
            data["generated_caption"] = generated_caption
            fout.write(json.dumps(data) + "\n")
            count += 1

        except Exception as e:
            print(f"⚠️  Skipping {image_path} due to error: {e}")

print(f"\n✅ Done! Generated captions for {count:,} samples.")
print(f"📁 Output saved to: {output_jsonl}")