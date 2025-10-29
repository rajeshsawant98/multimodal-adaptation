# tests/smoke/vqa_stub_smoke.py
import os
from datasets import load_dataset, Dataset, DownloadConfig

SKIP_HEAVY = os.getenv("SKIP_HEAVY_DOWNLOADS", "1") == "1"

def load_vqav2_small():
    if SKIP_HEAVY:
        # Fast, zero-download toy sample for CI/smoke.
        toy = {
            "question": ["What animal is in the picture?", "What color is the ball?"],
            "image": [
                "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/3/3a/Red_ball.png",
            ],
            "answers": [{"text": ["cat"]}, {"text": ["red"]}],
        }
        return Dataset.from_dict(toy)

    # Light(er) path: stream VQAv2 metadata and iterate just a few rows
    ds = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="validation",
        trust_remote_code=True,
        streaming=True,  # don't try to download / extract 13.5GB upfront
        download_config=DownloadConfig(max_retries=2, timeout=120),
    )
    # Take a tiny slice to prove the loader works without slurping the world
    return ds.take(5)

ds = load_vqav2_small()
first = next(iter(ds))
print("✅ VQA smoke OK | sample:", first["question"], "| answers:", first["answers"]["text"] if "answers" in first else first.get("answers"))