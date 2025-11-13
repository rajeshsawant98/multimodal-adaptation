import json
from collections import Counter, defaultdict
import re
import os

# The file you already generated
PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR", "/scratch/rsawant5/shared/shared_preprocessed")
INPUT = os.path.join(PREPROCESSED_DIR, "vqa_train.jsonl")

print("🔍 Analyzing:", INPUT)

def guess_type(q: str) -> str:
    q = q.lower().strip()

    # Yes/No
    if re.match(r"^(is|are|was|were|does|do|can|could|would|should)\b", q):
        return "yes/no"

    # Counting
    if q.startswith("how many"):
        return "count"

    # Color
    if q.startswith("what color"):
        return "color"

    # Location
    if q.startswith("where "):
        return "location"

    # Person
    if q.startswith("who "):
        return "who"

    # Type / Kind
    if q.startswith("what kind") or q.startswith("what type") or q.startswith("what sort"):
        return "type"

    # Time
    if q.startswith("when "):
        return "time"

    # Why
    if q.startswith("why "):
        return "reason"

    # Fallback
    return "other"


counts = Counter()
examples = defaultdict(list)

with open(INPUT, "r") as f:
    for line in f:
        row = json.loads(line)
        q = row["question"]
        t = guess_type(q)
        counts[t] += 1
        if len(examples[t]) < 5:
            examples[t].append(q)

print("\n===== QUESTION TYPE SUMMARY =====")
for t, c in counts.most_common():
    print(f"{t:10s} : {c}")

print("\n===== SAMPLE QUESTIONS PER TYPE =====")
for t in counts.keys():
    print(f"\n--- {t.upper()} ---")
    for ex in examples[t]:
        print("   ", ex)