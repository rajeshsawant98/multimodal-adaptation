# stage1_dataloading_fixed.py
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests
import pandas as pd

# ----------------------------
# Cache directory
# ----------------------------
cache_dir = Path("data/cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Utility to fetch images
# ----------------------------
def fetch_image(url: str):
    """Fetch an image from URL, fallback to None if fails."""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"[WARN] Could not fetch image: {e}")
        return None

def image_to_bytes(img):
    """Convert PIL image to PNG bytes for caching."""
    if img is None:
        return None
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

# ----------------------------
# Mini COCO
# ----------------------------
def load_coco_toy():
    toy_data = [
        {"image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"),
         "caption": "a cat sitting on the ground looking at the camera"},
        {"image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"),
         "caption": "a yellow labrador dog sitting on the grass"},
        {"image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/3/38/FakeSample.jpg"),
         "caption": "a bird flying in the sky"},
    ]
    # convert images to bytes
    for item in toy_data:
        item["image"] = image_to_bytes(item["image"])
    print(f"✅ Loaded {len(toy_data)} offline COCO samples.")
    return toy_data

# ----------------------------
# Mini Flickr30k
# ----------------------------
def load_flickr_toy():
    toy_data = [
        {"image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/7/70/FakeFlickr1.jpg"),
         "caption": "a man riding a bicycle down the street"},
        {"image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/0/0a/FakeFlickr2.jpg"),
         "caption": "a group of people sitting at a restaurant"},
        {"image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/9/9b/FakeFlickr3.jpg"),
         "caption": "a child playing with a ball in a park"},
    ]
    for item in toy_data:
        item["image"] = image_to_bytes(item["image"])
    print(f"✅ Loaded {len(toy_data)} offline Flickr30k samples.")
    return toy_data

# ----------------------------
# Mini VQAv2
# ----------------------------
def load_vqa_toy():
    toy_data = [
        {"image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"),
         "question": "What animal is this?", "answer": "cat"},
        {"image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"),
         "question": "What color is the dog?", "answer": "yellow"},
        {"image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/3/38/FakeSample.jpg"),
         "question": "What bird is shown?", "answer": "bird"},
    ]
    for item in toy_data:
        item["image"] = image_to_bytes(item["image"])
    print(f"✅ Loaded {len(toy_data)} offline VQAv2 samples.")
    return toy_data

# ----------------------------
# Save to cache
# ----------------------------
def save_cache(dataset, filename):
    df = pd.DataFrame(dataset)
    path = cache_dir / filename
    df.to_parquet(path)
    print(f"💾 Cached dataset to {path}")

# ----------------------------
# Clean old caches
# ----------------------------
def clean_old_caches():
    for file in cache_dir.glob("*.parquet"):
        file.unlink()
        print(f"🧹 Removed old cache: {file}")

# ----------------------------
# Main Stage 1 loader
# ----------------------------
if __name__ == "__main__":
    clean_old_caches()

    coco = load_coco_toy()
    flickr = load_flickr_toy()
    vqa = load_vqa_toy()

    save_cache(coco, "coco_mini.parquet")
    save_cache(flickr, "flickr_mini.parquet")
    save_cache(vqa, "vqa_toy.parquet")

    print("✅ Stage 1 datasets loaded, cached, and ready for BLIP-2 / CLIP experiments")
