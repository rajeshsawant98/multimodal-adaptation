# Multi-Modal Adaptation: In-Context Learning, Fine-Tuning, and Retrieval-Augmented Generation

### Overview
This repository contains an empirical comparison of three adaptation paradigms—In-Context Learning (ICL), parameter-efficient fine-tuning (e.g., LoRA/adapters), and Retrieval-Augmented Generation (RAG)—applied to multi-modal vision-language models (notably BLIP-2 and CLIP). The study evaluates these approaches on Image Captioning and Visual Question Answering (VQA) benchmarks (MS-COCO, Flickr30k, Visual Genome) and analyzes trade-offs among predictive performance, computational efficiency, and robustness to domain shift.

---

## Objectives

- Establish comparative baselines for In-Context Learning, parameter-efficient fine-tuning, and Retrieval-Augmented Generation in vision-language tasks.
- Quantify trade-offs in accuracy, computational cost, and robustness under domain shift.
- Provide reproducible experimental protocols and analysis to inform future research on multi-modal adaptation.

---

## Repository structure

```
multimodal-adaptation/
├── requirements.txt          # Core dependencies
├── README.md
├── tests/
│   └── smoke/                # Minimal sanity checks for major components
│       ├── run_all.py
│       ├── test_env.py       # verifies environment, Torch, CUDA/MPS
│       ├── faiss_smoke.py    # FAISS vector index and retrieval
│       ├── clip_embeddings_smoke.py  # CLIP image-text embeddings
│       ├── lora_peft_smoke.py        # LoRA/PEFT fine-tuning setup
│       ├── blip_caption_smoke.py     # BLIP-2 caption generation
│       └── vqa_stub_smoke.py         # VQAv2 integration test
├── src/                      # Training and evaluation modules
│   ├── datasets/             # COCO, Flickr30k, VQAv2 loaders
│   ├── models/               # BLIP-2, CLIP, PEFT wrappers
│   ├── rag/                  # Retrieval-Augmented Generation pipeline
│   ├── eval/                 # Metrics and evaluation scripts
│   └── utils/                # Helper functions and configuration
└── notebooks/                # Exploratory experiments and analysis
```

---

## Environment setup

1. Create a Python virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install required packages:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Notes:
- On Apple Silicon (M-series), PyTorch may employ the Metal (MPS) backend. On Linux systems with CUDA, GPU-enabled FAISS and other GPU-optimized libraries may be used when available.

---

## Libraries and dependencies

The experiments rely on standard machine learning and data-processing libraries. Representative versions (used in prior experiments) are listed below; exact versions may be adjusted to match the execution environment.

Core libraries
- torch (e.g., 2.4.0): Deep learning backend (MPS / CUDA / CPU)
- transformers (e.g., 4.44.2): Model implementations (CLIP, BLIP-2)
- datasets (e.g., 3.0.1): Dataset loading and preprocessing
- faiss-cpu / faiss-gpu (e.g., 1.8.0): Vector indexing for retrieval (RAG)
- peft (e.g., 0.10.x): Parameter-efficient fine-tuning (LoRA, adapters)

Vision and I/O
- Pillow (e.g., 10.x): Image loading and basic transformations
- opencv-python (optional): Advanced image operations and visualization

Text and tokenization
- sentencepiece (e.g., 0.2.x): Tokenization used by some models (BLIP-2/OPT)
- tokenizers (e.g., 0.19.x): Fast subword tokenizers

Evaluation and utilities
- pycocoevalcap: BLEU, ROUGE-L, CIDEr for captioning
- evaluate (e.g., 0.4.x): Hugging Face evaluation utilities
- pandas, numpy, tqdm: Data handling and progress reporting

Optional infrastructure
- wandb: Experiment tracking and visualization

Note: On Apple Silicon, PyTorch may use the MPS backend. On GPU-enabled Linux systems, GPU variants of FAISS and other libraries should be preferred when available.

---

## Running smoke tests

To perform minimal verification of the environment and core components, run the smoke tests:

```bash
cd tests/smoke
python run_all.py
```

A successful run reports that basic components are operational, including CLIP embeddings, BLIP-2 caption generation, LoRA fine-tuning plumbing, FAISS-based retrieval, and VQAv2 dataset integration.

---

## Model components

- CLIP (e.g., openai/clip-vit-base-patch32): Contrastive image-text encoder
- BLIP-2 (e.g., Salesforce/blip2-opt-2.7b): Vision–language model employing a query transformer
- LoRA / PEFT: Parameter-efficient fine-tuning techniques for adapting large models
- FAISS: Nearest-neighbor retrieval for RAG pipelines
- pycocoevalcap: Standard captioning metrics (BLEU, ROUGE-L, CIDEr)

---

## Planned experiments

The project roadmap outlines dataset preparation, baseline establishment for few-shot ICL, application of parameter-efficient fine-tuning and RAG integration, followed by cross-domain robustness evaluation and consolidation of results.

Representative phases
- Dataset setup and validation
- Few-shot ICL baselines for captioning and VQA
- LoRA fine-tuning and RAG integration experiments
- Cross-domain evaluation (e.g., COCO → Visual Genome)
- Consolidation of results and preparation of summary materials

---

## Evaluation metrics

- Captioning: BLEU-4, ROUGE-L, CIDEr (pycocoevalcap)
- VQA: Accuracy (by answer type), using standard VQAv2 evaluation procedures
- Efficiency: FLOPs, number of trainable parameters, latency (PyTorch profiler)
- Robustness: Cross-domain transfer experiments (e.g., COCO → Visual Genome)

---

## Known good environment

Representative versions used in experiments:
- Python 3.12.x
- PyTorch 2.4.0 (MPS or CUDA backends)
- Transformers 4.44.2
- datasets 3.0.1
- FAISS 1.8.0

## Datasets

The primary datasets considered in this work include MS-COCO, Flickr30k, VQAv2, and Visual Genome. Preprocessing is standardized across experiments (e.g., image resizing, text normalization) and tokenization is model-specific (SentencePiece for some models, BPE for others).

---

## Integration plan

Stages
- In-Context Learning: Evaluate zero-shot and few-shot performance for captioning and VQA
- Parameter-efficient fine-tuning: Apply LoRA/PEFT methods for task adaptation
- Retrieval-Augmented Generation: Integrate FAISS-based retrieval to provide context for generation
- Cross-domain evaluation: Assess generalization (e.g., COCO → Visual Genome)

---

For additional details, refer to the code modules under `src/` and the smoke tests under `tests/smoke`.
