"""
BLIP-2 OPT VQA ICL (Type-Aware, Clean Answers)
----------------------------------------------
- Uses BLIP-2 OPT-2.7B for VQA
- K-shot in-context learning with TYPE-AWARE sampling
- Few-shot pool = full vqa_train.jsonl
- Eval / debugging = vqa_debug.jsonl
- Stores both raw decoded answer and cleaned span after "Answer:"
- NO normalization of answers (you keep full text)

Run:
  python src/stage3_blip2_opt_vqa_icl_typeaware.py --K 0 --debug_limit 50
  python src/stage3_blip2_opt_vqa_icl_typeaware.py --K 1 --debug_limit 50
  python src/stage3_blip2_opt_vqa_icl_typeaware.py --K 3 --debug_limit 50
  python src/stage3_blip2_opt_vqa_icl_typeaware.py --K 5 --debug_limit 50
"""

import os
import json
import random
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -----------------------------------------------------
# Env + paths
# -----------------------------------------------------
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
PREPROCESSED_DIR = os.getenv("OUTPUT_JSONL_DIR")
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", PREPROCESSED_DIR)

OUT_DIR = os.path.join(EXPERIMENTS_DIR, "vqa_opt")
os.makedirs(OUT_DIR, exist_ok=True)

VQA_DEBUG_PATH = os.path.join(PREPROCESSED_DIR, "vqa_debug.jsonl")
VQA_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, "vqa_train.jsonl")

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------
# Model
# -----------------------------------------------------
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
print(f"\n🚀 Loading VQA model: {MODEL_NAME}")

processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device).eval()


# -----------------------------------------------------
# Question type detection (advanced-ish, inline version)
# -----------------------------------------------------

def detect_question_type(q: str) -> str:
    """
    Heuristic question-type classifier.
    Matches the main categories you saw in analyze_vqa_types_advanced.py:
      yes/no, count, color, object, type, location, relation, food, activity,
      animal, reason, sport, scene, who, material, brand, shape, time, weather, other
    """
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

    # reason / why
    if s.startswith("why "):
        return "reason"

    # who
    if s.startswith("who "):
        return "who"

    # type
    if s.startswith(("what type", "what kind")):
        return "type"

    # relation
    if any(p in s for p in [
        "next to", "in front of", "behind", "on top of", "under ", "above ",
        "beside", "between", "around"
    ]):
        return "relation"

    # shape
    if "shape" in s:
        return "shape"

    # food
    if any(w in s for w in ["pizza", "sandwich", "burger", "hot dog", "wine",
                            "beer", "food", "eat", "eating", "drinking", "cake"]):
        return "food"

    # sport
    if any(w in s for w in [
        "sport", "tennis", "baseball", "basketball", "soccer", "football",
        "skiing", "snowboard", "skateboard", "surfing"
    ]):
        return "sport"

    # animal
    if any(w in s for w in ["animal", "dog", "cat", "horse", "elephant",
                            "giraffe", "zebra", "bear", "cow", "sheep", "bird"]):
        return "animal"

    # material
    if "made of" in s or "material" in s:
        return "material"

    # brand / logo
    if any(w in s for w in ["brand", "logo", "company", "advertised"]):
        return "brand"

    # scene / room type
    if any(w in s for w in ["what room is this", "what kind of room", "which room"]):
        return "scene"

    # activity
    if "doing" in s or s.startswith("what is the man doing") or s.startswith("what is the woman doing"):
        return "activity"

    # explicit object identification
    if s.startswith("what is this") or s.startswith("what is the"):
        return "object"

    return "other"


# -----------------------------------------------------
# Few-shot ground-truth extraction
# -----------------------------------------------------

def extract_gt_from_sample(sample: dict) -> str:
    """
    Safely get a single ground-truth answer from any VQA v2-like record.
    """
    if "answer_gt" in sample:
        return sample["answer_gt"]
    if "answer" in sample:
        return sample["answer"]
    if "multiple_choice_answer" in sample:
        return sample["multiple_choice_answer"]
    if "answers" in sample and isinstance(sample["answers"], list) and len(sample["answers"]) > 0:
        cnt = Counter([x.get("answer", "") for x in sample["answers"]])
        return cnt.most_common(1)[0][0]
    return ""


# -----------------------------------------------------
# Load few-shot bank (full train set) and build type index
# -----------------------------------------------------

print(f"\n📂 Loading few-shot bank from: {VQA_TRAIN_PATH}")
fewshot_bank = [json.loads(l) for l in open(VQA_TRAIN_PATH)]
type_index = defaultdict(list)

for s in fewshot_bank:
    t = detect_question_type(s["question"])
    type_index[t].append(s)

print("✅ Few-shot bank loaded.")
print("   Types and counts:")
for t, lst in sorted(type_index.items(), key=lambda x: -len(x[1])):
    print(f"   {t:10s}: {len(lst)}")


# -----------------------------------------------------
# Few-shot block builder (TYPE AWARE)
# -----------------------------------------------------

def sample_few_shots(query_q: str, K: int):
    """
    Type-aware few-shot sampling.
    - Detect type of the query question
    - Sample K examples from that type
    - If not enough, fall back to global pool
    """
    if K <= 0:
        return []

    q_type = detect_question_type(query_q)
    pool = type_index.get(q_type, [])

    # if too few in that type, mix with full bank
    if len(pool) < K:
        pool = pool + fewshot_bank

    return random.sample(pool, K)


def build_examples_block_for_query(question: str, K: int) -> str:
    """
    Build a few-shot examples block for ICL.
    Format:
      ### Examples ###
      Q: ...
      A: ...
      Q: ...
      A: ...

    Or empty string if K == 0.
    """
    if K <= 0:
        return ""

    shots = sample_few_shots(question, K)
    lines = ["### Examples ###"]
    for s in shots:
        q = s["question"]
        a = extract_gt_from_sample(s)
        lines.append(f"Q: {q}")
        lines.append(f"A: {a}")
    return "\n".join(lines) + "\n\n"


# -----------------------------------------------------
# Prompt + answer cleaning
# -----------------------------------------------------

def make_prompt(question: str, examples_block: str) -> str:
    """
    Simple, stable prompt:
      [few-shot block if any]
      Question: ...
      Answer:
    """
    if examples_block:
        return f"{examples_block}Question: {question}\nAnswer:"
    else:
        return f"Question: {question}\nAnswer:"


def extract_answer_span(decoded: str) -> str:
    """
    From model decoded string, keep only the part after 'Answer:' if present.
    Otherwise, return the whole decoded string.
    We DO NOT normalize; we keep exact text.
    """
    text = decoded.strip()
    # Split on 'Answer:' (case-insensitive)
    lower = text.lower()
    key = "answer:"
    idx = lower.find(key)
    if idx != -1:
        return text[idx + len(key):].strip()
    return text


# -----------------------------------------------------
# Main ICL loop
# -----------------------------------------------------

def run_vqa_icl(K: int = 0, debug_limit: int | None = None):
    input_file = VQA_DEBUG_PATH
    output_file = os.path.join(OUT_DIR, f"vqa_opt_K{K}.jsonl")

    print(f"\n📂 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print(f"🔧 Mode: OPT VQA — K={K}")

    all_samples = [json.loads(l) for l in open(input_file)]
    fout = open(output_file, "w")

    for idx, sample in enumerate(tqdm(all_samples, desc=f"VQA K={K}")):
        if debug_limit is not None and idx >= debug_limit:
            break

        # Ground truth
        gt = extract_gt_from_sample(sample)

        # Image
        img_path = os.path.join(DATASETS_DIR, sample["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Few-shot block
        examples_block = build_examples_block_for_query(sample["question"], K)
        prompt = make_prompt(sample["question"], examples_block)

        # Encode
        inputs = processor(
            images=image,
            text=prompt,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate
        with torch.no_grad():
            out_tokens = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                num_beams=3,
            )

        decoded = processor.decode(out_tokens[0], skip_special_tokens=True).strip()
        cleaned = extract_answer_span(decoded)

        # Save
        fout.write(json.dumps({
            "image_path": sample["image_path"],
            "question": sample["question"],
            "answer_gt": gt,
            "answer_raw": decoded,
            f"answer_K{K}": cleaned,
        }) + "\n")

    fout.close()
    print(f"✅ DONE: {output_file}")


# -----------------------------------------------------
# CLI
# -----------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=0, help="number of ICL examples")
    parser.add_argument("--debug_limit", type=int, default=None, help="limit for quick tests")

    args = parser.parse_args()
    run_vqa_icl(K=args.K, debug_limit=args.debug_limit)