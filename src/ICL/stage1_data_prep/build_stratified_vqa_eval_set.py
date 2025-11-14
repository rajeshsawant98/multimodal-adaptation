"""
Build a STRATIFIED evaluation subset for VQAv2
----------------------------------------------
- Input: full vqa_train.jsonl (443k samples)
- Output: vqa_eval_stratified.jsonl (~3000 samples)
- Preserves the type distribution we computed
- Uses advanced question-type detection
"""

import os
import json
import random
from collections import defaultdict
from dotenv import load_dotenv

# -----------------------------------------------------
# Environment
# -----------------------------------------------------
load_dotenv()
PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR")
VQA_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, "vqa_train.jsonl")

OUT_PATH = os.path.join(PREPROCESSED_DIR, "vqa_eval_stratified.jsonl")
STATS_PATH = os.path.join(PREPROCESSED_DIR, "vqa_eval_stratified_stats.json")

print(f"\n📂 Loading full VQA train dataset: {VQA_TRAIN_PATH}")
rows = [json.loads(l) for l in open(VQA_TRAIN_PATH)]
print(f"✔ Loaded {len(rows):,} samples")

# -----------------------------------------------------
# Question type detection (same logic as ICL script)
# -----------------------------------------------------
def detect_question_type(q: str) -> str:
    s = q.strip().lower().rstrip(" ?.")

    # yes/no
    if s.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ",
                     "has ", "have ", "was ", "were ", "will ", "would ", "should ")):
        return "yes/no"

    # count
    if s.startswith(("how many", "how much", "number of")):
        return "count"

    # color
    if "color" in s or "colour" in s:
        return "color"

    # time
    if s.startswith("when ") or "time of day" in s:
        return "time"

    # weather
    if "weather" in s or "temperature" in s:
        return "weather"

    # location
    if s.startswith("where ") or "what city" in s or "what country" in s:
        return "location"

    # reason
    if s.startswith("why "):
        return "reason"

    # who
    if s.startswith("who "):
        return "who"

    # type
    if s.startswith(("what type", "what kind")):
        return "type"

    # relation
    if any(p in s for p in ["next to", "in front of", "behind", "on top of",
                            "under ", "above ", "beside", "between", "around"]):
        return "relation"

    # shape
    if "shape" in s:
        return "shape"

    # food
    if any(w in s for w in ["pizza","sandwich","burger","hot dog","wine","beer",
                            "food","eat","eating","drinking","cake"]):
        return "food"

    # sport
    if any(w in s for w in ["sport","tennis","baseball","basketball","soccer",
                            "football","ski","snowboard","skate","surf"]):
        return "sport"

    # animal
    if any(w in s for w in ["animal","dog","cat","horse","elephant","giraffe",
                            "zebra","bird","bear","cow","sheep"]):
        return "animal"

    # material
    if "made of" in s or "material" in s:
        return "material"

    # brand
    if any(w in s for w in ["brand","logo","company","advertis"]):
        return "brand"

    # scene (rooms)
    if any(w in s for w in ["what room is this", "what kind of room", "which room"]):
        return "scene"

    # activity
    if "doing" in s or s.startswith("what is the man doing") or s.startswith("what is the woman doing"):
        return "activity"

    # explicit object recognition
    if s.startswith("what is this") or s.startswith("what is the"):
        return "object"

    return "other"


# -----------------------------------------------------
# Target sample sizes (≈ 3000 total)
# -----------------------------------------------------
TARGET = {
    "yes/no": 300,
    "other": 300,
    "count": 300,
    "color": 300,
    "object": 200,
    "type": 200,
    "location": 200,
    "relation": 200,
    "food": 150,
    "activity": 150,
    "animal": 150,
    "reason": 100,
    "sport": 100,
    "scene": 100,
    "who": 100,
    "material": 60,
    "brand": 60,
    "shape": 60,
    "time": 60,
    "weather": 60,
}

print("\n🎯 Target per-type sample sizes:")
for t, sz in TARGET.items():
    print(f"  {t:10s}: {sz}")


# -----------------------------------------------------
# Bucket by type
# -----------------------------------------------------
print("\n📦 Bucketing samples by question type...")
buckets = defaultdict(list)

for r in rows:
    qtype = detect_question_type(r["question"])
    buckets[qtype].append(r)

print("✔ Bucketing done.\n")

# -----------------------------------------------------
# Sample stratified subset
# -----------------------------------------------------
print("🎯 Sampling stratified subset...")
final = []

for qtype, target_n in TARGET.items():
    candidates = buckets.get(qtype, [])
    if len(candidates) == 0:
        print(f"⚠ WARNING: No samples for type '{qtype}'")
        continue

    if len(candidates) < target_n:
        print(f"⚠ '{qtype}': only {len(candidates)} available → taking all.")
        chosen = candidates
    else:
        chosen = random.sample(candidates, target_n)

    final.extend(chosen)

random.shuffle(final)

print(f"✔ Final stratified set size: {len(final)} samples")

# -----------------------------------------------------
# Save outputs
# -----------------------------------------------------
with open(OUT_PATH, "w") as f:
    for x in final:
        f.write(json.dumps(x) + "\n")

with open(STATS_PATH, "w") as f:
    json.dump({
        "total": len(final),
        "per_type": {t: len([x for x in final if detect_question_type(x["question"]) == t])
                     for t in TARGET},
        "target": TARGET
    }, f, indent=2)

print(f"\n💾 Saved stratified evaluation set → {OUT_PATH}")
print(f"💾 Saved stats → {STATS_PATH}")
print("\n✅ DONE.")