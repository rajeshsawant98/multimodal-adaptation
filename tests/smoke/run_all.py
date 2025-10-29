import subprocess, sys, pathlib

import os, warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(
    "ignore",
    message=r"`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
    module=r"transformers\.tokenization_utils_base"
)

ROOT = pathlib.Path(__file__).parent
files = [
    "test_env.py",
    "faiss_smoke.py",
    "clip_embeddings_smoke.py",
    "lora_peft_smoke.py",
    "blip_caption_smoke.py",
    "vqa_stub_smoke.py",
]
for f in files:
    print("\n=== RUN:", f, "===")
    r = subprocess.run([sys.executable, str(ROOT / f)])
    if r.returncode != 0:
        print("❌ FAILED:", f); sys.exit(r.returncode)
print("\n✅ ALL SMOKE TESTS PASSED")