"""Gradio chat UI for sink-emergence nanoVLM checkpoints. Pick an arm + checkpoint step,
load it (pulling from the HF dataset on demand if not already local), then chat with an
(image, prompt) pair. Reuses generate.py's loader so the architecture always matches the
checkpoint (VLMConfig rebuilt from the run's run_config.json).

Usage:
  .venv/bin/python webchat.py [--port 7860] [--share]
"""

import argparse
import glob
import os
import re

import gradio as gr

from data.processors import get_image_processor, get_tokenizer
from generate import load_from_ckpt, pick_device

REPO_ID = "nemesismaniac/vlm-sink-emergence-ckpts"
RUNS_DIR = "runs"

_state = {}
_hf_files_cache = None


def list_arms():
    run_configs = sorted(glob.glob(os.path.join(RUNS_DIR, "*", "run_config.json")))
    return [os.path.basename(os.path.dirname(p)) for p in run_configs]


def _hf_files():
    global _hf_files_cache
    if _hf_files_cache is None:
        from huggingface_hub import HfApi
        _hf_files_cache = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    return _hf_files_cache


def list_steps(arm):
    """Checkpoint steps for an arm: union of what's on the HF dataset and what's local
    already (so already-downloaded ckpts still show up if the HF call fails offline)."""
    try:
        remote = _hf_files()
    except Exception:
        remote = []
    remote_pat = re.compile(rf"^runs/{re.escape(arm)}/ckpt_step(\d+)\.pt$")
    steps = {int(m.group(1)) for f in remote if (m := remote_pat.match(f))}
    local_pat = re.compile(r"ckpt_step(\d+)\.pt$")
    for f in glob.glob(os.path.join(RUNS_DIR, arm, "ckpt_step*.pt")):
        m = local_pat.search(f)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps, reverse=True)


def ensure_ckpt(arm, step):
    local_path = os.path.join(RUNS_DIR, arm, f"ckpt_step{step}.pt")
    if os.path.exists(local_path):
        return local_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, f"runs/{arm}/ckpt_step{step}.pt", repo_type="dataset")


def on_arm_change(arm):
    steps = list_steps(arm)
    return gr.Dropdown(choices=steps, value=steps[0] if steps else None)


def load_checkpoint(arm, step):
    if not arm or step is None:
        return "Pick an arm and step first."
    ckpt_path = ensure_ckpt(arm, int(step))
    run_config_path = os.path.join(RUNS_DIR, arm, 'run_config.json')
    device = pick_device()
    model = load_from_ckpt(ckpt_path, run_config_path).to(device)
    model.eval()
    _state.clear()
    _state.update(
        model=model,
        tokenizer=get_tokenizer(model.cfg.lm_tokenizer),
        image_processor=get_image_processor(model.cfg.vit_img_size),
        device=device,
    )
    return f"Loaded {arm} @ step {step} on {device}."


def respond(image, prompt, max_new_tokens, history):
    history = history or []
    if 'model' not in _state:
        return history + [{'role': 'assistant', 'content': 'Load a checkpoint first.'}], prompt
    if image is None:
        return history + [{'role': 'assistant', 'content': 'Upload an image first.'}], prompt

    model, tokenizer = _state['model'], _state['tokenizer']
    image_processor, device = _state['image_processor'], _state['device']

    template = f"Question: {prompt} Answer:"
    input_ids = tokenizer([template], return_tensors='pt')['input_ids'].to(device)
    img_t = image_processor(image.convert('RGB')).unsqueeze(0).to(device)
    gen = model.generate(input_ids, img_t, max_new_tokens=int(max_new_tokens))
    out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

    history = history + [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': out}]
    return history, ''


def build_app():
    arms = list_arms()
    default_arm = arms[0] if arms else None
    default_steps = list_steps(default_arm) if default_arm else []

    with gr.Blocks(title='nanoVLM sink-emergence chat') as demo:
        gr.Markdown('# Talk to a sink-emergence nanoVLM checkpoint')
        with gr.Row():
            arm_dd = gr.Dropdown(arms, value=default_arm, label='Arm')
            step_dd = gr.Dropdown(default_steps, value=default_steps[0] if default_steps else None,
                                   label='Checkpoint step')
            load_btn = gr.Button('Load model')
        status = gr.Textbox(label='Status', interactive=False)

        arm_dd.change(on_arm_change, inputs=arm_dd, outputs=step_dd)
        load_btn.click(load_checkpoint, inputs=[arm_dd, step_dd], outputs=status)

        with gr.Row():
            image_in = gr.Image(type='pil', label='Image', value='assets/image.png')
            with gr.Column():
                chatbot = gr.Chatbot(label='Chat', height=400)
                prompt_in = gr.Textbox(label='Prompt', placeholder='What is this?')
                max_tokens = gr.Slider(5, 100, value=30, step=5, label='Max new tokens')
                with gr.Row():
                    send_btn = gr.Button('Send')
                    clear_btn = gr.Button('Clear')

        send_btn.click(respond, inputs=[image_in, prompt_in, max_tokens, chatbot], outputs=[chatbot, prompt_in])
        prompt_in.submit(respond, inputs=[image_in, prompt_in, max_tokens, chatbot], outputs=[chatbot, prompt_in])
        clear_btn.click(lambda: [], outputs=chatbot)

    return demo


def main():
    p = argparse.ArgumentParser(description='Gradio chat UI for sink-emergence nanoVLM checkpoints.')
    p.add_argument('--port', type=int, default=7860)
    p.add_argument('--share', action='store_true')
    args = p.parse_args()
    build_app().launch(server_name='0.0.0.0', server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
