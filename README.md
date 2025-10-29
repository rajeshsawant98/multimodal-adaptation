# 🧠 Multi-Modal Adaptation: In-Context Learning vs Fine-Tuning vs Retrieval-Augmented Generation

### 📘 Overview
This repository implements a comparative study of **three major adaptation paradigms** —  
**In-Context Learning (ICL)**, **Fine-Tuning**, and **Retrieval-Augmented Generation (RAG)** —  
within **multi-modal vision-language models** such as **BLIP-2** and **CLIP**.  

We benchmark their performance on **Image Captioning** and **Visual Question Answering (VQA)** tasks  
across datasets like **MS-COCO**, **Flickr30k**, and **Visual Genome**, analyzing the trade-offs between  
**accuracy**, **efficiency**, and **robustness**.

---

## 🎯 Objectives

- Benchmark ICL, parameter-efficient fine-tuning (LoRA/adapters), and RAG for vision-language tasks.  
- Evaluate trade-offs in **accuracy**, **compute cost**, and **robustness** under domain shift.  
- Provide reproducible baselines and insights for future multi-modal adaptation research.

---

## 🧩 Repository Structure

```
multimodal-adaptation/
├── requirements.txt          # Core dependencies
├── README.md                 # You are here
│
├── tests/
│   └── smoke/                # Minimal sanity checks for all components
│       ├── run_all.py
│       ├── test_env.py       # verifies environment, Torch, CUDA/MPS
│       ├── faiss_smoke.py    # FAISS vector index + retrieval
│       ├── clip_embeddings_smoke.py  # CLIP image-text embeddings
│       ├── lora_peft_smoke.py        # LoRA/PEFT fine-tuning setup
│       ├── blip_caption_smoke.py     # BLIP-2 image captioning
│       └── vqa_stub_smoke.py         # VQAv2 mini test
│
├── src/                      # (planned) training/evaluation modules
│   ├── datasets/             # COCO, Flickr30k, VQAv2 loaders
│   ├── models/               # BLIP-2, CLIP, PEFT wrappers
│   ├── rag/                  # Retrieval-Augmented Generation pipeline
│   ├── eval/                 # Metrics and evaluation scripts
│   └── utils/                # Helpers and config
│
└── notebooks/                # exploratory experiments
```

---

## ⚙️ Environment Setup

### 1️⃣ Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

💡 On macOS (M-series), PyTorch uses the **Metal (MPS)** backend automatically.  
💡 On Linux with CUDA, FAISS-GPU and bitsandbytes will be used automatically.

---

## 📦 Libraries and Dependencies

| Category | Library | Version | Purpose |
|-----------|----------|----------|----------|
| **Core ML** | `torch` | 2.4.0 | Deep learning backend (MPS / CUDA / CPU) |
|  | `transformers` | 4.44.2 | Hugging Face models (CLIP, BLIP-2, etc.) |
|  | `datasets` | 3.0.1 | Dataset loading and preprocessing (VQAv2, COCO) |
|  | `faiss-cpu` / `faiss-gpu` | 1.8.0 | Vector indexing for retrieval (RAG) |
|  | `peft` | 0.10.x | Parameter-Efficient Fine-Tuning (LoRA, adapters) |
| **Vision / Image I/O** | `Pillow` | 10.x | Image loading and transformations |
|  | `opencv-python` | 4.x | Optional — advanced image ops and visualization |
| **Text / Tokenization** | `sentencepiece` | 0.2.x | Tokenization for BLIP-2 / OPT models |
|  | `tokenizers` | 0.19.x | Fast subword tokenization (BPE / WordPiece) |
| **Evaluation** | `pycocoevalcap` | latest | BLEU, ROUGE-L, CIDEr metrics for captioning |
|  | `evaluate` | 0.4.x | Unified evaluation wrapper for HF metrics |
| **Data & Utils** | `pandas` | 2.x | Data manipulation |
|  | `numpy` | 1.26.x | Numerical operations |
|  | `tqdm` | 4.x | Progress bars |
| **Experiment Tracking (Optional)** | `wandb` | 0.17.x | Experiment logging & visualization |
| **I/O / Networking** | `requests` | 2.32.x | Robust HTTP requests for image/data download |
|  | `aiohttp` | 3.x | Async I/O (used internally by datasets) |
| **System / Warnings** | `certifi` | latest | SSL cert verification |
|  | `packaging` | latest | Version handling for model dependencies |

### 💡 Notes

- On Apple Silicon, `torch.backends.mps` provides native GPU acceleration.  
- `sitecustomize.py` silences tokenizer warnings:
  ```python
  import warnings
  warnings.filterwarnings("ignore", message=r"`clean_up_tokenization_spaces`", category=FutureWarning)
  ```

---

## 🧪 Running Smoke Tests

Run all smoke tests to verify environment and dependencies:

```bash
cd tests/smoke
python run_all.py
```

If successful, you’ll see:

```
✅ ALL SMOKE TESTS PASSED
```

This validates:
- CLIP image-text embeddings  
- BLIP-2 caption generation  
- LoRA fine-tuning pipeline  
- FAISS vector retrieval  
- VQAv2 dataset integration  

---

## 🧬 Model Components

| Component | Purpose | Library |
|------------|----------|----------|
| **CLIP** | Contrastive Image-Text Pretraining | `openai/clip-vit-base-patch32` |
| **BLIP-2** | Vision-Language model w/ Query Transformer | `Salesforce/blip2-opt-2.7b` |
| **LoRA / PEFT** | Parameter-Efficient Fine-Tuning | `peft` |
| **FAISS** | Fast nearest-neighbor retrieval for RAG | `faiss-cpu/faiss-gpu` |
| **PyCOCOEvalCap** | Captioning metrics (BLEU, ROUGE, CIDEr) | `pycocoevalcap` |

---

## 📊 Planned Experiments

| Phase | Description | Deliverables |
|-------|--------------|--------------|
| **Weeks 1-2** | Dataset setup, literature review, smoke testing | verified pipelines |
| **Weeks 3-4** | Few-shot ICL baselines (captioning/VQA) | accuracy + qualitative results |
| **Weeks 5-6** | LoRA fine-tuning & RAG integration | efficiency curves |
| **Week 7** | Cross-domain evaluation (COCO → Visual Genome) | robustness metrics |
| **Week 8** | Consolidate results + final report & slides | comparative analysis |

---

## 🧠 Evaluation Metrics

| Task | Metric | Tool |
|------|---------|------|
| **Captioning** | BLEU-4, ROUGE-L, CIDEr | `pycocoevalcap` |
| **VQA** | Accuracy by answer type | `VQAv2` evaluator |
| **Efficiency** | FLOPs, trainable params, latency | PyTorch profiler |
| **Robustness** | COCO → VG domain transfer | custom scripts |

---

## ⚡ Known Good Environment

| Component | Version |
|------------|----------|
| Python | 3.12.x |
| Torch | 2.4.0 (MPS or CUDA) |
| Transformers | 4.44.2 |
| Datasets | 3.0.1 |
| FAISS | 1.8.0 |
| BLIP-2 | 2.7B model |
| CLIP | ViT-B/32 |

---
