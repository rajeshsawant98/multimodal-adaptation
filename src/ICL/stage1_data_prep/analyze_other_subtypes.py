import json
import os
import re
from collections import Counter, defaultdict

PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR", "/scratch/rsawant5/shared/shared_preprocessed")
INPUT = os.path.join(PREPROCESSED_DIR, "vqa_train.jsonl")

print("🔍 Analyzing OTHER CATEGORY:", INPUT)

# --------------------------------------------------------
# Subtype rules inside "other"
# --------------------------------------------------------

def detect_other_subtype(q):
    q = q.lower().strip()

    # Object name queries
    if re.match(r"what is this\b", q) or re.match(r"what is the\b", q):
        return "object"

    # What is the person holding / wearing / using
    if "holding" in q:
        return "holding"
    if "wearing" in q:
        return "wearing"
    if "using" in q:
        return "using"

    # What is the person doing
    if "doing" in q:
        return "activity"

    # What is on / in / next to → spatial relation
    if re.match(r"what (is )?(on|in|at|next to|behind|beside|under|over)", q):
        return "relation"

    # Animals (but didn’t match animal category earlier)
    if any(word in q for word in ["dog", "cat", "bird", "horse", "elephant", "bear", "zebra", "giraffe"]):
        return "animal-context"

    # Clothing queries
    if "shirt" in q or "hat" in q or "pants" in q or "dress" in q:
        return "clothing"

    # Tool / instrument
    if any(x in q for x in ["tool", "racket", "bat", "camera", "phone", "knife", "fork"]):
        return "tool"

    # Vehicle-related
    if any(x in q for x in ["car", "train", "bus", "truck", "motorcycle", "plane"]):
        return "vehicle"

    # Body parts
    if any(x in q for x in ["hand", "arm", "leg", "head", "face", "hair"]):
        return "body-part"

    # Artifacts / structure
    if any(x in q for x in ["building", "tower", "bridge"]):
        return "structure"

    # Measurement / scale
    if "size" in q or "distance" in q or "long" in q or "big" in q:
        return "measurement"

    # General attribute queries
    if "made of" in q or "material" in q or "brand" in q:
        return "attribute"

    # Wild “what” questions we didn’t classify
    if q.startswith("what"):
        return "general-what"

    return "misc"


# --------------------------------------------------------
# Load data and run subtype analysis
# --------------------------------------------------------

from analyze_vqa_types_advanced import detect_type  # reuse main classifier

counts = Counter()
examples = defaultdict(list)

with open(INPUT) as f:
    for line in f:
        row = json.loads(line)
        q = row["question"]

        # Only analyze those originally classified as OTHER
        if detect_type(q) != "other":
            continue

        subtype = detect_other_subtype(q)
        counts[subtype] += 1

        if len(examples[subtype]) < 5:
            examples[subtype].append(q)

# --------------------------------------------------------
# Display results
# --------------------------------------------------------

print("\n===== OTHER — SUBTYPE DISTRIBUTION =====")
for t, c in counts.most_common():
    print(f"{t:15s}: {c}")

print("\n===== SAMPLE QUESTIONS PER SUBTYPE =====")
for t in counts:
    print(f"\n--- {t.upper()} ---")
    for q in examples[t]:
        print("   ", q)