from datasets import load_dataset

def load_flickr_small(split="test", num_samples=10):
    """
    Loads a small subset of the Flickr30k captions dataset for smoke testing.
    Automatically limits to `num_samples` and works cross-platform.
    """
    print("[INFO] Using lightweight dataset: nlphuji/flickr30k")

    try:
        ds = load_dataset("nlphuji/flickr30k", split=f"{split}[:{num_samples}]")
    except Exception as e:
        print(f"[WARN] Failed to load nlphuji/flickr30k: {e}")
        print("[INFO] Falling back to toy examples.")
        return [
            {"caption": "A cat sitting on the sofa.", "image": None},
            {"caption": "A man riding a bike near the lake.", "image": None},
        ]

    records = []
    for ex in ds:
        img = ex.get("image", None)
        caption = ex.get("caption", "")

        # Handle either dict (URL) or PIL.Image.Image
        if isinstance(img, dict) and "url" in img:
            img = img["url"]

        records.append({"image": img, "caption": caption})

    print(f"✅ Loaded {len(records)} Flickr30k samples successfully.")
    return records