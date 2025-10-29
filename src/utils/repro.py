import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_dtype():
    if torch.cuda.is_available(): 
        return "cuda", torch.float16
    if torch.backends.mps.is_available(): 
        return "mps", torch.float16  # Apple Silicon
    return "cpu", torch.float32