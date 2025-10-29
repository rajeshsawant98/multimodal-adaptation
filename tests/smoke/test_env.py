import torch, transformers, faiss, importlib.util
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), "| MPS:", torch.backends.mps.is_available())
print("Transformers:", transformers.__version__)
print("FAISS present:", hasattr(faiss, "IndexFlatIP"))
print("OpenCLIP present:", importlib.util.find_spec("open_clip_torch") is not None)
print("✅ ENV OK")