# Multi-Modal Adaptation: In-Context Learning, Fine-Tuning, and Retrieval-Augmented Generation

### Overview
This repository contains an empirical comparison of three adaptation paradigms—In-Context Learning (ICL), parameter-efficient fine-tuning (e.g., LoRA/adapters), and Retrieval-Augmented Generation (RAG)—applied to multi-modal vision-language models (notably BLIP-2 and CLIP). The study evaluates these approaches on Image Captioning and Visual Question Answering (VQA) benchmarks (MS-COCO, Flickr30k, Visual Genome) and analyzes trade-offs among predictive performance, computational efficiency, and robustness to domain shift.

---

## Objectives

- Establish comparative baselines for In-Context Learning, parameter-efficient fine-tuning, and Retrieval-Augmented Generation in vision-language tasks.
- Quantify trade-offs in accuracy, computational cost, and robustness under domain shift.
- Provide reproducible experimental protocols and analysis to inform future research on multi-modal adaptation.

---


## In-Context Learning (ICL) using BLIP-2 OPT 2.7b
This repository provides a structured framework for evaluating multi-modal adaptation methods—specifically **In-Context Learning (ICL)**—using state-of-the-art vision–language models such as **BLIP-2 OPT 2.7B**. Current efforts focus on **Visual Question Answering (VQA)** using systematic preprocessing, stratified sampling, type-aware ICL prompting, and multi-level answer evaluation (exact, normalized, fuzzy).

The project is modular and designed for reproducibility, scalability on HPC systems (ASU SOL), and clean extensibility toward fine-tuning and RAG-based pipelines.

---

## Table of Contents
- [Overview](#overview)  
- [Repository Structure](#repository-structure)  
- [Environment Setup](#environment-setup)  
- [Datasets](#datasets)  
- [Data Preprocessing](#data-preprocessing)  
- [VQA Stratified Evaluation](#vqa-stratified-evaluation)  
- [In-Context Learning (BLIP-2 OPT)](#in-context-learning-blip-2-opt)  
- [Evaluation Pipeline](#evaluation-pipeline)  
- [Results Summary](#results-summary)  
- [Reproducing Experiments](#reproducing-experiments)  
- [Citation](#citation)

---

## Repository Structure

```
multimodal-adaptation/
│
├── src/
│   ├── ICL/
│   │   ├── stage1_data_prep/
│   │   │   ├── analyze_vqa_types_advanced.py
│   │   │   ├── build_stratified_vqa_eval_set.py
│   │   │   ├── generate_vqa_testset.py
│   │   │   └── validate_jsonl_datasets.py
│   │   ├── stage2_baseline/
│   │   ├── stage3_vqa_blip2/
│   │   │   └── stage3_blip2_opt_vqa_icl_typeaware.py
│   │   └── stage4_evaluation/
│   │       └── eval_vqa_stratified_full.py
│   │
│   ├── datasets/
│   ├── models/
│   ├── rag/
│   ├── utils/
│
├── notebooks/
├── tests/smoke/
└── README.md
```

---

## Environment Setup

```
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Recommended versions:

- Python ≥ 3.10  
- PyTorch ≥ 2.1 (CUDA or MPS)  
- Transformers ≥ 4.44  
- SentencePiece, Pillow, tqdm, pandas  

---

## Datasets

### VQAv2

Expected directory structure:

```
DATASETS_DIR/
└── VQAv2/
    ├── train2014/
    ├── val2014/
    ├── annotations/
    └── questions/
```

The preprocessing scripts convert these into unified **JSONL** files with aligned fields:

- `vqa_train.jsonl`
- `vqa_val.jsonl`
- `vqa_debug.jsonl`
- `vqa_eval_stratified.jsonl`

---

## Data Preprocessing

Provided under:

```
src/ICL/stage1_data_prep/
```

### Core scripts

| Script | Purpose |
|--------|---------|
| `analyze_vqa_types_advanced.py` | Extracts VQA question types using rule-based classifier with 20+ categories. |
| `build_stratified_vqa_eval_set.py` | Creates a *balanced 3150-sample evaluation split* with fixed counts per question type. |
| `validate_jsonl_datasets.py` | Ensures JSONL consistency, missing images, schema validation. |

To generate the stratified eval set:

```
python src/ICL/stage1_data_prep/build_stratified_vqa_eval_set.py     --train_jsonl shared_preprocessed/vqa_train.jsonl     --val_jsonl shared_preprocessed/vqa_val.jsonl     --out_file shared_preprocessed/vqa_eval_stratified.jsonl
```

---

## VQA Stratified Evaluation

We use a **balanced evaluation set**:  
- 3150 total samples  
- 21 question types × 150 samples each  

This avoids dataset bias (e.g., yes/no dominating other categories).

---

## In-Context Learning (BLIP-2 OPT)

Path:

```
src/ICL/stage3_vqa_blip2/stage3_blip2_opt_vqa_icl_typeaware.py
```

Key features:

- BLIP-2 OPT-2.7B model  
- Type-aware few-shot example selection  
- Clean answer extraction  
- Deterministic decoding (beam search, no sampling)  
- No normalization during inference (evaluation handles that)  

Run example:

```
python src/ICL/stage3_vqa_blip2/stage3_blip2_opt_vqa_icl_typeaware.py     --K 3     --eval_file vqa_eval_stratified.jsonl     --out_suffix strat_K3
```

---

## Evaluation Pipeline

Script:

```
src/ICL/stage4_evaluation/eval_vqa_stratified_full.py
```

Supports:

- Exact match  
- Normalized match (lowercase, remove articles/punctuation)  
- Fuzzy match (Levenshtein ≥ 90%)  
- Per-type grouped metrics  
- CSV + JSON output  

Run:

```
python eval_vqa_stratified_full.py   --gt_file shared_preprocessed/vqa_eval_stratified.jsonl   --pred_files vqa_opt_strat_K0.jsonl vqa_opt_strat_K1.jsonl ...   --out_prefix vqa_opt_strat
```

---

## Results Summary (BLIP-2 OPT)

| K-shot | Accuracy (Combined) |
|--------|---------------------|
| 0 | 0.131 |
| 1 | 0.264 |
| 3 | **0.299** |
| 5 | 0.289 |

Notes:

- Improvement saturates around K=3  
- Some types remain challenging (reasoning, location)  
- Color, yes/no, and scene questions improve significantly with ICL  

---

## Reproducing Experiments

### Step 1 — Preprocess datasets
```
python src/ICL/stage1_data_prep/validate_jsonl_datasets.py
python src/ICL/stage1_data_prep/build_stratified_vqa_eval_set.py
```

### Step 2 — Run BLIP-2 OPT ICL
```
python src/ICL/stage3_vqa_blip2/stage3_blip2_opt_vqa_icl_typeaware.py --K 3 --eval_file vqa_eval_stratified.jsonl
```

### Step 3 — Evaluate predictions
```
python src/ICL/stage4_evaluation/eval_vqa_stratified_full.py   --gt_file vqa_eval_stratified.jsonl   --pred_files vqa_opt_strat_K0.jsonl ...
```

---

## Citation

If you use this repository in academic or research work:

```
@software{multimodal_adaptation_2025,
  title={Multi-Modal Adaptation Framework: Vision--Language ICL and Evaluation},
  author={Sawant, Rajesh A. and contributors},
  year={2025},
  url={https://github.com/...}
}
```

---

## Contributions

Contributions, issues, and feature requests are welcome.  
Please open a Pull Request or Issue on GitHub.

