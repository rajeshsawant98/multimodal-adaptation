"""
Full VQA Stratified Evaluation (Exact + Normalized + Fuzzy)
-----------------------------------------------------------
- Evaluates BLIP-2 OPT VQA predictions on a stratified subset
- Uses the same heuristic question-type detector as the ICL script
- Supports multiple K-shot result files in one run

Metrics per K:
  - exact_raw:     pred == gt (as-is)
  - exact_norm:    normalized_pred == normalized_gt
  - fuzzy_match:   fuzzy ratio(norm_pred, norm_gt) >= 0.9
  - combined:      exact_raw OR exact_norm OR fuzzy_match

Outputs:
  - <out_prefix>_results.json
  - <out_prefix>_results.csv

Example:
  python src/eval/eval_vqa_stratified_full.py \
    --gt_file /scratch/rsawant5/shared/shared_preprocessed/vqa_eval_stratified.jsonl \
    --pred_files \
      /scratch/rsawant5/shared/all_experiments/vqa_opt/vqa_opt_strat_K0.jsonl \
      /scratch/rsawant5/shared/all_experiments/vqa_opt/vqa_opt_strat_K1.jsonl \
      /scratch/rsawant5/shared/all_experiments/vqa_opt/vqa_opt_strat_K3.jsonl \
      /scratch/rsawant5/shared/all_experiments/vqa_opt/vqa_opt_strat_K5.jsonl \
    --out_prefix vqa_opt_strat_eval
"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Dict, Tuple, List


# -----------------------------------------------------
# Question type detection (same as in ICL script)
# -----------------------------------------------------
def detect_question_type(q: str) -> str:
    s = q.strip().lower().rstrip(" ?.")

    if s.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ",
                     "has ", "have ", "was ", "were ", "will ", "would ", "should ")):
        return "yes/no"

    if s.startswith(("how many", "how much", "number of")):
        return "count"

    if "color" in s or "colour" in s:
        return "color"

    if s.startswith("when ") or "time of day" in s:
        return "time"

    if "weather" in s or "temperature" in s:
        return "weather"

    if s.startswith("where ") or "what city" in s or "what country" in s:
        return "location"

    if s.startswith("why "):
        return "reason"

    if s.startswith("who "):
        return "who"

    if s.startswith(("what type", "what kind")):
        return "type"

    if any(p in s for p in [
        "next to", "in front of", "behind", "on top of", "under ", "above ",
        "beside", "between", "around"
    ]):
        return "relation"

    if "shape" in s:
        return "shape"

    if any(w in s for w in [
        "pizza", "sandwich", "burger", "hot dog", "wine",
        "beer", "food", "eat", "eating", "drinking", "cake"
    ]):
        return "food"

    if any(w in s for w in [
        "sport", "tennis", "baseball", "basketball", "soccer", "football",
        "skiing", "snowboard", "skateboard", "surfing"
    ]):
        return "sport"

    if any(w in s for w in [
        "animal", "dog", "cat", "horse", "elephant",
        "giraffe", "zebra", "bear", "cow", "sheep", "bird"
    ]):
        return "animal"

    if "made of" in s or "material" in s:
        return "material"

    if any(w in s for w in ["brand", "logo", "company", "advertised"]):
        return "brand"

    if any(w in s for w in ["what room is this", "what kind of room", "which room"]):
        return "scene"

    if "doing" in s:
        return "activity"

    if s.startswith("what is this") or s.startswith("what is the"):
        return "object"

    return "other"


# -----------------------------------------------------
# Normalization + fuzzy matching
# -----------------------------------------------------
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""

    text = ans.strip().lower()

    # remove punctuation
    text = _PUNCT_RE.sub("", text)

    # remove articles
    text = _ARTICLE_RE.sub(" ", text)

    # map common yes/no variants
    if text in {"y", "ya", "yeah", "yep"}:
        text = "yes"
    if text in {"n", "nope"}:
        text = "no"

    # collapse whitespace
    text = " ".join(text.split())

    # naive singularization for last token (cats -> cat)
    tokens = text.split()
    if len(tokens) == 1 and tokens[0].endswith("s") and len(tokens[0]) > 3:
        tokens[0] = tokens[0][:-1]
        text = tokens[0]

    return text


def fuzzy_equal(a: str, b: str, threshold: float = 0.9) -> bool:
    if not a and not b:
        return True
    if not a or not b:
        return False
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


# -----------------------------------------------------
# Data structures
# -----------------------------------------------------
@dataclass
class MatchStats:
    total: int = 0
    exact_raw: int = 0
    exact_norm: int = 0
    fuzzy: int = 0
    combined: int = 0

    def to_dict(self):
        if self.total == 0:
            acc_exact = acc_norm = acc_fuzzy = acc_combined = 0.0
        else:
            acc_exact = self.exact_raw / self.total
            acc_norm = self.exact_norm / self.total
            acc_fuzzy = self.fuzzy / self.total
            acc_combined = self.combined / self.total

        d = asdict(self)
        d.update({
            "acc_exact_raw": acc_exact,
            "acc_exact_norm": acc_norm,
            "acc_fuzzy": acc_fuzzy,
            "acc_combined": acc_combined,
        })
        return d


# -----------------------------------------------------
# Loading helpers
# -----------------------------------------------------
def load_gt_file(gt_path: str) -> Dict[Tuple[str, str], dict]:
    """
    Load stratified GT file into a dict keyed by (image_path, question).
    Adds question_type field.
    """
    gt_index = {}
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img = row["image_path"]
            q = row["question"]
            gt = row.get("answer_gt") or row.get("answer") or row.get("multiple_choice_answer", "")
            qtype = detect_question_type(q)
            key = (img, q)
            gt_index[key] = {
                "image_path": img,
                "question": q,
                "answer_gt": gt,
                "qtype": qtype,
            }
    return gt_index


def infer_K_label(path: str) -> str:
    """
    Try to infer K (0,1,3,5,...) from filename like vqa_opt_strat_K3.jsonl.
    If not found, use the basename without extension.
    """
    base = os.path.basename(path)
    m = re.search(r"_K(\d+)", base)
    if m:
        return f"K{m.group(1)}"
    # fallback: whole basename
    return os.path.splitext(base)[0]


# -----------------------------------------------------
# Evaluation core
# -----------------------------------------------------
def evaluate_predictions_for_file(
    gt_index: Dict[Tuple[str, str], dict],
    pred_path: str,
) -> Dict[str, MatchStats]:
    """
    Returns a dict:
      'GLOBAL' -> MatchStats
      '<qtype>' -> MatchStats per question type
    """
    stats_global = MatchStats()
    stats_by_type: Dict[str, MatchStats] = defaultdict(MatchStats)

    missing_gt = 0

    with open(pred_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img = row["image_path"]
            q = row["question"]

            # prediction field: use any key that starts with 'answer_K'
            pred = None
            for k, v in row.items():
                if k.startswith("answer_K"):
                    pred = v
                    break
            # fallback: try 'answer_raw'
            if pred is None:
                pred = row.get("answer_raw", "")

            key = (img, q)
            if key not in gt_index:
                missing_gt += 1
                continue

            gt_row = gt_index[key]
            gt = gt_row["answer_gt"]
            qtype = gt_row["qtype"]

            # compute matches
            gt_raw = (gt or "").strip()
            pred_raw = (pred or "").strip()

            exact_raw = int(gt_raw == pred_raw)

            gt_norm = normalize_answer(gt_raw)
            pred_norm = normalize_answer(pred_raw)

            exact_norm = int(gt_norm == pred_norm)
            fuzzy = int(fuzzy_equal(gt_norm, pred_norm, threshold=0.9))
            combined = int(bool(exact_raw or exact_norm or fuzzy))

            # update stats
            stats_global.total += 1
            stats_global.exact_raw += exact_raw
            stats_global.exact_norm += exact_norm
            stats_global.fuzzy += fuzzy
            stats_global.combined += combined

            st = stats_by_type[qtype]
            st.total += 1
            st.exact_raw += exact_raw
            st.exact_norm += exact_norm
            st.fuzzy += fuzzy
            st.combined += combined

    if missing_gt > 0:
        print(f"[WARN] {missing_gt} predictions in {pred_path} had no matching GT entry.")

    results = {"GLOBAL": stats_global}
    results.update({qt: st for qt, st in stats_by_type.items()})
    return results


# -----------------------------------------------------
# Pretty printing helpers
# -----------------------------------------------------
def print_results_table(label: str, results: Dict[str, MatchStats]):
    print(f"\n===== RESULTS FOR {label} =====")
    global_stats = results["GLOBAL"].to_dict()
    print(f"Global:")
    print(f"  N               : {global_stats['total']}")
    print(f"  Exact (raw)     : {global_stats['acc_exact_raw']:.3f}")
    print(f"  Exact (norm)    : {global_stats['acc_exact_norm']:.3f}")
    print(f"  Fuzzy (>=0.9)   : {global_stats['acc_fuzzy']:.3f}")
    print(f"  Combined        : {global_stats['acc_combined']:.3f}")

    print("\nPer question type (combined acc):")
    header = f"{'qtype':12s} {'N':>6s} {'acc_exact':>10s} {'acc_norm':>10s} {'acc_fuzzy':>10s} {'acc_comb':>10s}"
    print(header)
    print("-" * len(header))

    for qtype, st in sorted(results.items()):
        if qtype == "GLOBAL":
            continue
        d = st.to_dict()
        print(
            f"{qtype:12s} "
            f"{d['total']:6d} "
            f"{d['acc_exact_raw']:10.3f} "
            f"{d['acc_exact_norm']:10.3f} "
            f"{d['acc_fuzzy']:10.3f} "
            f"{d['acc_combined']:10.3f}"
        )


def save_results_json_csv(
    all_results: Dict[str, Dict[str, MatchStats]],
    out_prefix: str,
):
    # JSON
    json_path = f"{out_prefix}_results.json"
    json_payload = {}
    for k_label, res in all_results.items():
        json_payload[k_label] = {
            qtype: st.to_dict() for qtype, st in res.items()
        }
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"\n💾 Saved JSON results to: {json_path}")

    # CSV (long format)
    csv_path = f"{out_prefix}_results.csv"
    with open(csv_path, "w") as f:
        f.write("K_label,qtype,total,exact_raw,exact_norm,fuzzy,combined,"
                "acc_exact_raw,acc_exact_norm,acc_fuzzy,acc_combined\n")
        for k_label, res in all_results.items():
            for qtype, st in res.items():
                d = st.to_dict()
                f.write(
                    f"{k_label},{qtype},"
                    f"{d['total']},{d['exact_raw']},{d['exact_norm']},{d['fuzzy']},{d['combined']},"
                    f"{d['acc_exact_raw']:.6f},{d['acc_exact_norm']:.6f},{d['acc_fuzzy']:.6f},{d['acc_combined']:.6f}\n"
                )
    print(f"💾 Saved CSV results to: {csv_path}")


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_file",
        type=str,
        required=True,
        help="Path to stratified GT jsonl file (e.g., vqa_eval_stratified.jsonl)",
    )
    parser.add_argument(
        "--pred_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more prediction jsonl files (e.g., vqa_opt_strat_K0.jsonl ...)",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="vqa_opt_strat",
        help="Prefix for output JSON/CSV files",
    )

    args = parser.parse_args()

    print(f"\n🔍 Loading GT from: {args.gt_file}")
    gt_index = load_gt_file(args.gt_file)
    print(f"  Loaded {len(gt_index)} GT entries.")

    all_results: Dict[str, Dict[str, MatchStats]] = {}

    for pred_path in args.pred_files:
        k_label = infer_K_label(pred_path)
        print(f"\n=== Evaluating file: {pred_path} (label={k_label}) ===")
        res = evaluate_predictions_for_file(gt_index, pred_path)
        all_results[k_label] = res
        print_results_table(k_label, res)

    save_results_json_csv(all_results, args.out_prefix)


if __name__ == "__main__":
    main()