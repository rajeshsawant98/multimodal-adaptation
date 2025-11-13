import json
import re
import os
from collections import Counter, defaultdict

PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR", "/scratch/rsawant5/shared/shared_preprocessed")
INPUT = os.path.join(PREPROCESSED_DIR, "vqa_train.jsonl")

print("🔍 Analyzing:", INPUT)

# --------------------------------------
# Improved category mapping
# --------------------------------------

def detect_type(q):
    q = q.lower().strip()

    # yes/no
    if re.match(r"^(is|are|was|were|do|does|did|can|could|should|would|has|have)\b", q):
        return "yes/no"

    # count
    if q.startswith("how many"):
        return "count"

    # color
    if "color" in q:
        return "color"

    # location
    if q.startswith("where"):
        return "location"

    # who / person
    if q.startswith("who"):
        return "who"

    # when / time
    if q.startswith("when"):
        return "time"

    # why / reason
    if q.startswith("why"):
        return "reason"

    # what is the person doing
    if re.match(r"what .* doing", q):
        return "activity"

    # what is on / in / next to → spatial relation
    if re.match(r"what (is )?(on|in|at|next to|behind|beside|under|over)", q):
        return "relation"

    # what animal
    if re.match(r"what (kind of )?(animal|dog|cat|bird|horse|bear|elephant)", q):
        return "animal"

    # what sport
    if "sport" in q:
        return "sport"

    # what brand / label
    if "brand" in q or "company" in q or "logo" in q:
        return "brand"

    # scene classification
    if "room" in q or "type of room" in q:
        return "scene"

    # food / eating
    if re.search(r"(what.*eat|food|dish|meal|pizza|burger|sandwich|drink)", q):
        return "food"

    # material
    if "made of" in q or "material" in q:
        return "material"

    # weather
    if "weather" in q:
        return "weather"

    # shape
    if "shape" in q:
        return "shape"

    # type / kind general
    if q.startswith("what kind") or q.startswith("what type") or q.startswith("what sort"):
        return "type"

    # object
    if q.startswith("what is the") or q.startswith("what is this"):
        return "object"

    # fallback
    return "other"

# --------------------------------------
# Scan dataset
# --------------------------------------

counts = Counter()
examples = defaultdict(list)

with open(INPUT) as f:
    for line in f:
        row = json.loads(line)
        q = row["question"]
        t = detect_type(q)
        counts[t] += 1
        if len(examples[t]) < 5:
            examples[t].append(q)

# --------------------------------------
# Display results
# --------------------------------------

print("\n===== ADVANCED QUESTION TYPE SUMMARY =====")
for t, c in counts.most_common():
    print(f"{t:12s}: {c}")

print("\n===== SAMPLE QUESTIONS PER TYPE =====")
for t in counts:
    print(f"\n--- {t.upper()} ---")
    for q in examples[t]:
        print("   ", q)