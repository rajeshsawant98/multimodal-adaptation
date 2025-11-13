import os
import json
import random
from dotenv import load_dotenv

# ------------------------------
# Load environment
# ------------------------------
load_dotenv()

PRE_DIR = os.getenv("OUTPUT_JSONL_DIR")     # /scratch/rsawant5/shared/shared_preprocessed
if PRE_DIR is None:
    raise ValueError("❌ OUTPUT_JSONL_DIR not set in .env")

src = os.path.join(PRE_DIR, "vqa_train.jsonl")
dst = os.path.join(PRE_DIR, "vqa_debug.jsonl")

# ------------------------------
# Parameters
# ------------------------------
SAMPLE_SIZE = 200   # change to 50 or 100 as needed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ------------------------------
# Load the source dataset
# ------------------------------
if not os.path.exists(src):
    raise FileNotFoundError(f"❌ Source VQA file not found: {src}")

print(f"📘 Loading: {src}")

with open(src, "r", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]

total = len(rows)
print(f"📊 Total records available: {total}")

# ------------------------------
# Sample smaller subset
# ------------------------------
sample_size = min(SAMPLE_SIZE, total)
small = random.sample(rows, sample_size)

# ------------------------------
# Write output file
# ------------------------------
print(f"✍️ Writing debug file: {dst}")

with open(dst, "w", encoding="utf-8") as f:
    for x in small:
        f.write(json.dumps(x) + "\n")

print(f"✅ Created {dst} with {sample_size} examples")