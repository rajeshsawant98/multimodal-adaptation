"""
Unified DataLoader for captioning and VQA tasks.
Reads standardized .jsonl files produced in Stage 0 preprocessing.

Supports:
  - COCO / Flickr30k captioning datasets
  - VQAv2 question-answering dataset
"""

import os
import json
from typing import Dict, Any, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dotenv import load_dotenv

# ----------- ENV VARIABLES -----------
load_dotenv()
DATASETS_DIR = os.getenv("DATASETS_DIR")
OUTPUT_JSONL_DIR = os.getenv("OUTPUT_JSONL_DIR")

# ----------- IMAGE TRANSFORMS -----------
DEFAULT_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----------- DATASET CLASS -----------
class JsonlDataset(Dataset):
    """
    Loads multimodal samples from a JSONL file.

    For captioning:
      {"image_path": "...", "caption": "..."}

    For VQA:
      {"image_path": "...", "question": "...", "answer": "..."}
    """
    def __init__(self,
                 jsonl_path: str,
                 datasets_dir: str = DATASETS_DIR,
                 transform=None,
                 task: str = "auto",
                 limit: int = None):
        self.jsonl_path = jsonl_path
        self.datasets_dir = datasets_dir
        self.transform = transform or DEFAULT_IMAGE_TRANSFORM

        # Load JSONL lines
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Auto-detect task type
        if task == "auto":
            sample = self.samples[0]
            self.task = "vqa" if {"question", "answer"}.issubset(sample.keys()) else "caption"
        else:
            self.task = task

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.datasets_dir, sample["image_path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.task == "caption":
            return {
                "image": image,
                "caption": sample["caption"],
                "image_path": sample["image_path"]
            }
        else:
            return {
                "image": image,
                "question": sample["question"],
                "answer": sample["answer"],
                "image_path": sample["image_path"]
            }

# ----------- DATALOADER WRAPPER -----------
def make_dataloader(file_name: str,
                    batch_size: int = 16,
                    shuffle: bool = True,
                    limit: int = None,
                    num_workers: int = 4) -> DataLoader:
    """
    Returns a DataLoader for the given JSONL file name.
    Auto-detects task type (captioning or VQA).
    """
    jsonl_path = os.path.join(OUTPUT_JSONL_DIR, file_name)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"❌ JSONL file not found: {jsonl_path}")

    dataset = JsonlDataset(jsonl_path, DATASETS_DIR, task="auto", limit=limit)
    task = dataset.task
    print(f"📦 Loaded {len(dataset):,} samples from {file_name} ({task})")

    def collate_fn(batch):
        images = torch.stack([b["image"] for b in batch])
        if task == "caption":
            texts = [b["caption"] for b in batch]
            return {"images": images, "captions": texts}
        else:
            questions = [b["question"] for b in batch]
            answers = [b["answer"] for b in batch]
            return {"images": images, "questions": questions, "answers": answers}

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=collate_fn)

# ----------- SMOKE TEST -----------
if __name__ == "__main__":
    print("\n🚀 Running quick data loading sanity check...")

    # Captioning sample (COCO/Flickr)
    cap_loader = make_dataloader("captions_combined.jsonl", batch_size=4, limit=8)
    cap_batch = next(iter(cap_loader))
    print("🖼️ Caption batch:", cap_batch["images"].shape, "Captions:", cap_batch["captions"][:2])

    # VQA sample
    vqa_loader = make_dataloader("vqa_val.jsonl", batch_size=4, limit=8)
    vqa_batch = next(iter(vqa_loader))
    print("💬 VQA batch:", vqa_batch["images"].shape, "Questions:", vqa_batch["questions"][:2])