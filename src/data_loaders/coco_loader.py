from PIL import Image
from io import BytesIO
import requests

def fetch_image(url: str):
    """Fetch image from URL, fallback to None if fails."""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"[WARN] Could not fetch image: {e}")
        return None

def load_coco_toy():
    """Offline mini COCO subset for Stage 1 experiments."""
    toy_data = [
        {
            "image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"),
            "caption": "a cat sitting on the ground looking at the camera",
        },
        {
            "image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"),
            "caption": "a yellow labrador dog sitting on the grass",
        },
        {
            "image": fetch_image("https://upload.wikimedia.org/wikipedia/commons/5/54/Blackbird_%28Turdus_merula%29_%281%29.jpg"),
            "caption": "a blackbird standing on a rock outdoors",
        },
        # Add more inline samples here (total 10–50)
    ]
    print(f"✅ Loaded {len(toy_data)} offline COCO samples.")
    return toy_data

if __name__ == "__main__":
    data = load_coco_toy()
    print("Sample caption:", data[0]["caption"])
