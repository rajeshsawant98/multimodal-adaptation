from transformers import (
    CLIPProcessor,
    CLIPModel,
    CLIPImageProcessor,
    CLIPTokenizerFast,
    logging as hf_logging,
)
from PIL import Image
from io import BytesIO
import torch, requests

# (optional) quiet transformers warnings
hf_logging.set_verbosity_error()

# --- device & dtype ---
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
use_fp16 = device.type in {"mps", "cuda"}  # safe half on accelerators

# --- load model/processor ---
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device, dtype=torch.float16 if use_fp16 else torch.float32).eval()
tokenizer = CLIPTokenizerFast.from_pretrained(
    "openai/clip-vit-base-patch32",
    clean_up_tokenization_spaces=True,
)
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)

# --- robust image fetch ---
url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, allow_redirects=True)
resp.raise_for_status()
img = Image.open(BytesIO(resp.content)).convert("RGB")

# --- inputs ---
inputs = proc(text=["a cat", "a dog"], images=img, return_tensors="pt", padding=True)
# move to device + dtype
inputs = {k: v.to(device) for k, v in inputs.items()}
if use_fp16:
    # only cast float tensors; token ids should stay int
    inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}

# --- inference ---
with torch.inference_mode():
    logits = model(**inputs).logits_per_image.softmax(dim=-1).tolist()[0]

print("✅ CLIP probs [cat, dog]:", [round(p, 3) for p in logits])
