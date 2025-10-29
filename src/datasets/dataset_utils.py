from PIL import Image
import requests
from io import BytesIO

def fetch_image(url: str):
    """Safely download an image and return a PIL object."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to fetch image from {url}: {e}")
        return None