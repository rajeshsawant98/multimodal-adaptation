from huggingface_hub import hf_hub_download  # optional, if you use it
from datasets import load_dataset, DownloadConfig  # <-- use full module name
from PIL import Image
from io import BytesIO
import requests


def fetch_image(url: str):
    """Safely fetch an image from URL."""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"[WARN] Could not fetch image from {url}: {e}")
        return None


def load_coco_small(split: str = "train", num_samples: int = 10):
    """
    Loads a small subset of the MS-COCO 2017 captioning dataset.
    Attempts HuggingFaceM4 mirrors first, falls back to toy samples if offline.
    """

    tried = []
    dataset = None
    candidates = [
        "HuggingFaceM4/COCO",
        "HuggingFaceM4/coco_captions",
        "nyu-mll/coco",
    ]

    cfg = DownloadConfig(max_retries=5, resume_download=True, num_proc=1)

    for name in candidates:
        try:
            print(f"[INFO] trying {name}")
            dataset = load_dataset(
                name,
                split=f"{split}[:{num_samples}]",
                trust_remote_code=True,
                download_config=cfg,
            )
            print(f"✅ Loaded from {name}")
            break
        except Exception as e:
            print(f"[WARN] failed {name}: {e}")
            tried.append(name)

    if dataset is None:
        print("[INFO] Falling back to toy inline dataset.")
        return [
            {
                "image": fetch_image(
                    "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
                ),
                "caption": "a cat sitting on the ground looking at the camera",
            },
            {
                "image": fetch_image(
                    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
                ),
                "caption": "a yellow labrador dog sitting on the grass",
            },
        ]

    records = []
    for row in dataset:
        img = None
        if "image" in row:
            if isinstance(row["image"], dict) and "url" in row["image"]:
                img = fetch_image(row["image"]["url"])
            elif isinstance(row["image"], Image.Image):
                img = row["image"]
        caption = row.get("caption", row.get("text", ""))
        records.append({"image": img, "caption": caption})

    print(f"✅ Loaded {len(records)} COCO samples successfully.")
    return records


if __name__ == "__main__":
    data = load_coco_small(num_samples=2)
    print("Sample caption:", data[0]["caption"])