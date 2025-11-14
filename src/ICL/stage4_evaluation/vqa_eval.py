"""
VQA Evaluation Script
----------------------
Evaluates BLIP-2 / InstructBLIP VQA predictions.

Supports:
✓ Exact Match
✓ Soft Match (token normalized)
✓ VQA v2 Official Accuracy
✓ Question-Type Breakdown
✓ CSV + JSON Summary
"""

import os
import json
import re
import string
from collections import defaultdict, Counter
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------
# Load environment
# ---------------------------------------------
load_dotenv()
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR")
OUT_DIR = os.path.join(EXPERIMENTS_DIR, "vqa_opt")


# ---------------------------------------------
# Utility Functions
# ---------------------------------------------

def normalize(text: str) -> str:
    """
    Soft normalization:
    - lowercase
    - remove punctuation
    - strip spaces
    - collapse repeated spaces
    """
    if text is None:
        return ""

    text = text.lower()
    text = text.strip()
    text = text.replace(",", "")

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def vqa_score(pred, gt_list):
    """
    Implements VQAv2 accuracy:
    score = min(1, (# humans who answered pred) / 3)
    For your dataset, gt may be string or list of answer dicts.
    """

    if isinstance(gt_list, str):
        # treat single answer as list with 10 identical votes
        gt_list = [{"answer": gt_list}] * 10

    answers = [a["answer"] for a in gt_list]

    votes = sum([normalize(pred) == normalize(a) for a in answers])
    return min(1.0, votes / 3.0)


def detect_qtype(q):
    """Simple heuristic for grouping question types."""
    q_lower = q.lower()

    if q_lower.startswith("is ") or q_lower.startswith("are ") or q_lower.startswith("do "):
        return "yes/no"
    if "how many" in q_lower:
        return "count"
    if "what color" in q_lower:
        return "color"
    if "where" in q_lower:
        return "location"
    if "why" in q_lower:
        return "reason"
    if "who" in q_lower:
        return "who"
    if "what type" in q_lower or "what kind" in q_lower:
        return "type"

    return "other"


# ---------------------------------------------
# Evaluation Function
# ---------------------------------------------
def evaluate_file(path):
    print(f"\n🔍 Evaluating: {path}")

    rows = [json.loads(l) for l in open(path)]
    results = []

    for row in rows:
        gt = row["answer_gt"]
        raw_pred = row["answer_raw"]
        pred = row[list(row.keys())[-1]]  # answer_K0 or answer_K3 etc.

        # save outputs
        results.append({
            "image_path": row["image_path"],
            "question": row["question"],
            "qtype": detect_qtype(row["question"]),
            "gt": gt,
            "pred_raw": raw_pred,
            "pred": pred,
            "exact": int(pred.strip().lower() == gt.lower()),
            "soft": int(normalize(pred) == normalize(gt)),
            "vqa_acc": vqa_score(pred, [{"answer": gt}]*10),
        })

    df = pd.DataFrame(results)

    # ---- aggregate summary ----
    summary = {
        "num_samples": len(df),
        "exact_match": df["exact"].mean(),
        "soft_match": df["soft"].mean(),
        "vqa_accuracy": df["vqa_acc"].mean(),
    }

    # ---- per question type summary ----
    type_summary = (
        df.groupby("qtype")[["exact", "soft", "vqa_acc"]]
          .mean()
          .reset_index()
          .to_dict(orient="records")
    )

    print("\n===== GLOBAL METRICS =====")
    print(json.dumps(summary, indent=2))

    print("\n===== PER QUESTION TYPE =====")
    for t in type_summary:
        print(t)

    # ---- save reports ----
    base = os.path.basename(path).replace(".jsonl", "")
    df.to_csv(os.path.join(OUT_DIR, base + "_eval.csv"), index=False)

    json.dump(
        {"summary": summary, "type_breakdown": type_summary},
        open(os.path.join(OUT_DIR, base + "_eval.json"), "w"),
        indent=2
    )

    print(f"\n💾 Saved CSV + JSON eval for {base}")
    return df


# ---------------------------------------------
# CLI
# ---------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True,
                        help="Path to vqa_opt_K*.jsonl")

    args = parser.parse_args()
    evaluate_file(args.file)