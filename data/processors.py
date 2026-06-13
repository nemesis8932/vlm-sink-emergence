from transformers import AutoTokenizer

TOKENIZERS_CACHE = {}

def get_tokenizer(name):
    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]

def get_image_processor(img_size):
    # torchvision is the training/remote path; fall back to an equivalent pure-PIL processor
    # (bilinear resize + CHW float in [0,1]) when torchvision is unavailable (e.g. Mac smoke test).
    try:
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    except ImportError:
        return _PILImageProcessor(img_size)


class _PILImageProcessor:
    """torchvision-free Resize+ToTensor (top-level so it pickles across DataLoader workers)."""
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img):
        import numpy as np
        import torch
        from PIL import Image
        img = img.convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
