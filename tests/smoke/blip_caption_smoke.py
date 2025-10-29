from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BlipForConditionalGeneration,
    BlipImageProcessor,
    AutoTokenizer,
)
from PIL import Image
from io import BytesIO
import torch, requests

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
print("Device:", device, "| dtype:", dtype)

# reliable image
url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
resp.raise_for_status()
img = Image.open(BytesIO(resp.content)).convert("RGB")

# lightweight BLIP-2
#name = "Salesforce/blip2-flan-t5-xl"
name = "Salesforce/blip-image-captioning-base"
tokenizer = AutoTokenizer.from_pretrained(name, clean_up_tokenization_spaces=True)
image_processor = BlipImageProcessor.from_pretrained(name)
processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
model = BlipForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(device)
#model = Blip2ForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(device)

# prepare inputs: move tensors to device and cast floating tensors to the desired dtype
inputs = processor(images=img, return_tensors="pt")
inputs = {
    k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
    for k, v in inputs.items()
}
out = model.generate(**inputs, max_new_tokens=20 )
caption = processor.decode(out[0], skip_special_tokens=True )
print("✅ BLIP-2 Caption:", caption)
