# stage2_iclModels.py (fixed for varying column names)
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------
# Step 1: Load Stage 1 cached datasets
# ----------------------------
coco = pd.read_parquet("data/cache/coco_mini.parquet")
flickr = pd.read_parquet("data/cache/flickr_mini.parquet")
vqa = pd.read_parquet("data/cache/vqa_toy.parquet")

print(f"✅ Loaded {len(coco)} samples from coco_mini.parquet")
print(f"✅ Loaded {len(flickr)} samples from flickr_mini.parquet")
print(f"✅ Loaded {len(vqa)} samples from vqa_toy.parquet")

# ----------------------------
# Step 2: Load small model for few-shot ICL
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
print(f"✅ Loaded {model_name} on {device}")

# ----------------------------
# Step 3: Prepare few-shot prompt function
# ----------------------------
def generate_caption(dataset_name, sample_text_list, new_text_desc):
    prompt = f"Dataset: {dataset_name}\nFew-shot examples:\n"
    for i, txt in enumerate(sample_text_list, 1):
        prompt += f"{i}. {txt}\n"
    prompt += f"Generate a similar caption/answer for: '{new_text_desc}'"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# ----------------------------
# Step 4: Run few-shot caption generation for each dataset
# ----------------------------
num_few_shot = 2
print("\n=== Few-shot ICL Outputs ===\n")

datasets = [("COCO-mini", coco), ("Flickr-mini", flickr), ("VQAv2-toy", vqa)]

for dataset_name, df in datasets:
    # Pick text column dynamically
    for col in ["caption", "question", "text"]:
        if col in df.columns:
            text_col = col
            break
    else:
        print(f"[WARN] No suitable text column found in {dataset_name}, skipping...")
        continue

    sample_texts = df[text_col][:num_few_shot].tolist()
    new_text_desc = "An image of a dog playing with a ball in a park."
    generated_caption = generate_caption(dataset_name, sample_texts, new_text_desc)
    print(f"[{dataset_name}] Generated caption:")
    print(generated_caption)
    print("-" * 50)

print("✅ Stage 2 ICL processing complete")
