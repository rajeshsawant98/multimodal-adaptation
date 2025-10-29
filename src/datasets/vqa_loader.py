from datasets import load_dataset, DownloadConfig


def load_vqa_small(split: str = "validation", num_samples: int = 10):
    """
    Loads a small subset of the Visual Question Answering v2 dataset.
    If full download fails, falls back to toy examples.
    """

    cfg = DownloadConfig(max_retries=5, resume_download=True, num_proc=1)

    try:
        print("[INFO] Trying full VQAv2 load from HuggingFaceM4/VQAv2")
        ds = load_dataset(
            "HuggingFaceM4/VQAv2",
            split=f"{split}[:{num_samples}]",
            trust_remote_code=True,
            download_config=cfg,
        )
        print(f"✅ Loaded {len(ds)} VQAv2 samples successfully.")
        return [
            {
                "question": row["question"],
                "answers": [a["text"] for a in row.get("answers", [])],
                "image": row.get("image", None),
            }
            for row in ds
        ]

    except Exception as e:
        print(f"[WARN] VQAv2 load failed: {e}")
        print("[INFO] Falling back to toy inline dataset.")
        return [
            {
                "question": "What animal is shown?",
                "answers": ["cat"],
                "image": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
            },
            {
                "question": "What color is the bus?",
                "answers": ["yellow"],
                "image": "https://upload.wikimedia.org/wikipedia/commons/7/7e/YellowSchoolBus.jpg",
            },
        ]


if __name__ == "__main__":
    data = load_vqa_small(num_samples=2)
    print("Sample:", data[0])