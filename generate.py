"""Generate text from an (image, prompt) pair with a sink-emergence nanoVLM.

Two load paths:
  --ckpt <runs/<arm>/ckpt_stepN.pt>   raw bf16 state-dict from train_sinks.save_ckpt();
                                       VLMConfig is rebuilt from the sibling run_config.json
                                       (its "vlm_cfg" key already encodes the arm's
                                       lm_attn_gate/lm_attn_impl, so the architecture matches
                                       the checkpoint without needing --arm/--vit_init).
  --pretrained <dir_or_hf_repo>        save_pretrained()/HF hub dir; VisionLanguageModel.from_pretrained.

Default (no flags) pulls the upstream lusxvr/nanoVLM-222M for a quick "does this work" check.
"""

import argparse
import json
import os

import torch
from PIL import Image

from data.processors import get_image_processor, get_tokenizer
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel

torch.manual_seed(0)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(args):
    if args.ckpt:
        run_config_path = args.run_config or os.path.join(os.path.dirname(args.ckpt), 'run_config.json')
        with open(run_config_path) as f:
            run_config = json.load(f)
        cfg = VLMConfig(**run_config['vlm_cfg'])
        model = VisionLanguageModel(cfg, load_backbone=False)
        sd = torch.load(args.ckpt, map_location='cpu')
        sd = {k: v.float() for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, f"{args.ckpt}: missing={missing} unexpected={unexpected}"
        print(f"[load] {args.ckpt} (cfg from {run_config_path})", flush=True)
        return model

    print(f"[load] from_pretrained {args.pretrained}", flush=True)
    return VisionLanguageModel.from_pretrained(args.pretrained)


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from an image with a sink-emergence nanoVLM.")
    src = p.add_mutually_exclusive_group()
    src.add_argument('--ckpt', type=str, default=None,
                      help="Raw bf16 state-dict from train_sinks.py, e.g. runs/baseline/ckpt_step18287.pt")
    src.add_argument('--pretrained', type=str, default='lusxvr/nanoVLM-222M',
                      help="Local save_pretrained() dir or HF repo id (default: upstream nanoVLM-222M)")
    p.add_argument('--run_config', type=str, default=None,
                    help="run_config.json for --ckpt (default: sibling of --ckpt)")
    p.add_argument('--image', type=str, nargs='+', default=['assets/image.png'], help="One or more image paths")
    p.add_argument('--prompt', type=str, default='What is this?')
    p.add_argument('--generations', type=int, default=1, help="Sampled generations per image")
    p.add_argument('--max_new_tokens', type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    print(f"Using device: {device}", flush=True)

    model = load_model(args).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    template = f"Question: {args.prompt} Answer:"
    input_ids = tokenizer([template] * len(args.image), return_tensors='pt')['input_ids'].to(device)
    images = torch.stack([image_processor(Image.open(p).convert('RGB')) for p in args.image]).to(device)

    print(f"\nPrompt: {args.prompt}\n")
    for g in range(args.generations):
        gen = model.generate(input_ids, images, max_new_tokens=args.max_new_tokens)
        outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for img_path, out in zip(args.image, outs):
            print(f"[gen {g + 1}] {img_path}: {out}")


if __name__ == '__main__':
    main()
