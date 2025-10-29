from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from peft.mapping import get_peft_model
import torch

tok = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
tok.clean_up_tokenization_spaces = True

mdl = AutoModelForCausalLM.from_pretrained("gpt2")

cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM" , fan_in_fan_out=True, )

mdl = get_peft_model(mdl, cfg)

trainable = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
total = sum(p.numel() for p in mdl.parameters())
print(f"✅ LoRA wired | trainable={trainable:,} / total={total:,}")

inp = tok("Hello, my name is", return_tensors="pt")
out = mdl.generate(**inp, max_new_tokens=10)
print("Sample:", tok.decode(out[0], skip_special_tokens=True))
