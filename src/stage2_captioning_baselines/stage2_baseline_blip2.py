"""
Generates captions for images using BLIP-2 (Flan-T5 XL or OPT-2.7B fallback).
Later we’ll extend this for few-shot / ICL prompting.
"""

import os, json, torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -------- Load environment paths --------
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
OUTPUT_JSONL_DIR = os.getenv("OUTPUT_JSONL_DIR")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Select BLIP-2 model variant --------
print("🚀 Selecting appropriate BLIP-2 model...")
try:
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
except Exception:
    gpu_mem = 0

# Use Flan-T5-XL if GPU has >= 24 GB, else fallback to OPT-2.7B
# if gpu_mem >= 24:
#     model_name = "Salesforce/blip2-flan-t5-xl"
# else:
#     model_name = "Salesforce/blip2-opt-2.7b"

model_name = "Salesforce/blip2-flan-t5-xl"

print(f"📦 Using model: {model_name}  (GPU memory detected: {gpu_mem:.1f} GB)")

# -------- Load BLIP-2 model --------
print(f"🚀 Loading BLIP-2 model...")
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device).eval()
print(f"✅ Model loaded on {device}")

# -------- Input / Output files --------
input_jsonl  = os.path.join(OUTPUT_JSONL_DIR, "captions_debug.jsonl")
output_jsonl = os.path.join(OUTPUT_JSONL_DIR, "blip2_baseline_preds.jsonl")

print(f"📂 Reading from: {input_jsonl}")
print(f"💾 Saving predictions to: {output_jsonl}")

# -------- Caption generation --------
count = 0
with open(input_jsonl, "r") as fin, open(output_jsonl, "w") as fout:
    for line in tqdm(fin, desc="Generating captions", ncols=100):
        data = json.loads(line)
        img_path = os.path.join(DATASETS_DIR, data["image_path"])
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            # Simple descriptive prompt (can be tuned later for ICL)
            prompt = "Describe the image in one complete sentence."

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=40)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            data["generated_caption"] = caption
            fout.write(json.dumps(data) + "\n")
            count += 1

        except Exception as e:
            print(f"⚠️  Skipping {img_path} due to error: {e}")

print(f"\n✅ Finished! Generated captions for {count:,} samples.")
print(f"📁 Results saved at: {output_jsonl}")