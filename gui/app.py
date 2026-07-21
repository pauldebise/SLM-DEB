#!/usr/bin/env python3
"""
Gradio inference GUI for SLM checkpoints.
Lists available checkpoints, loads model from embedded config,
and provides a text generation interface.

Usage:
  python gui/app.py [--checkpoint-dir checkpoints] [--tokenizer data/tokenizer/tokenizer.json]
  python gui/app.py --port 7860 --share
"""

import argparse
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

import gradio as gr
import torch
import yaml

from model.transformer import Transformer, TransformerConfig


def list_checkpoints(checkpoint_dir: str) -> list:
    if not os.path.isdir(checkpoint_dir):
        return []
    ckpts = []
    for f in sorted(os.listdir(checkpoint_dir)):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            path = os.path.join(checkpoint_dir, f)
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                step = ckpt.get("step", 0)
                cfg = ckpt.get("config", {})
                name = cfg.get("name", "unknown")
                n_params = sum(
                    v.numel() for v in ckpt.get("model_state_dict", {}).values()
                )
                label = f"step={step:07d} | {name} | {n_params/1e6:.1f}M params"
                ckpts.append((label, path))
            except Exception as e:
                ckpts.append((f"{f} (error: {e})", None))
    return ckpts


def load_model_for_inference(checkpoint_path: str, device: str = "cuda"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = ckpt.get("config", {})
    state_dict = ckpt.get("model_state_dict", {})

    if not config_dict:
        raise ValueError("Checkpoint does not contain model config")

    tcfg = TransformerConfig(config_dict)
    model = Transformer(tcfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config_dict


class InferenceSession:
    def __init__(self, tokenizer_path: str = None):
        self.model = None
        self.config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
            except Exception:
                pass

    def load_checkpoint(self, checkpoint_path: str):
        if not checkpoint_path or checkpoint_path == "None":
            return "No checkpoint selected"
        try:
            self.model, self.config = load_model_for_inference(checkpoint_path, self.device)
            return (f"Loaded: {self.config.get('name', 'unknown')} | "
                    f"{self.config.get('n_layers')} layers, "
                    f"d_model={self.config.get('d_model')} | "
                    f"on {self.device}")
        except Exception as e:
            return f"Error loading checkpoint: {e}"

    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.9):
        if self.model is None:
            return "No model loaded. Select a checkpoint first."

        if self.tokenizer:
            encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        else:
            input_ids = torch.tensor(
                [[ord(c) % 32000 + 1 for c in prompt[-512:]]],
                dtype=torch.long, device=self.device,
            )

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if top_p < 1.0 else None,
            )

        if self.tokenizer:
            new_ids = output_ids[0, input_ids.size(1):].tolist()
            return self.tokenizer.decode(new_ids, skip_special_tokens=False)
        else:
            return "[No tokenizer available — showing raw token IDs]\n" + \
                   str(output_ids[0].tolist())


def refresh_checkpoints(checkpoint_dir: str):
    ckpts = list_checkpoints(checkpoint_dir)
    if not ckpts:
        return gr.Dropdown(choices=[], value=None)
    choices = [c[0] for c in ckpts]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def build_interface(session: InferenceSession, checkpoint_dir: str):
    ckpts = list_checkpoints(checkpoint_dir)
    choices = [c[0] for c in ckpts]
    default_choice = choices[0] if choices else None

    with gr.Blocks(title="SLM Trainer — Inference") as app:
        gr.Markdown("# SLM Trainer — Inference")

        with gr.Row():
            with gr.Column(scale=1):
                ckpt_dropdown = gr.Dropdown(
                    choices=choices,
                    value=default_choice,
                    label="Checkpoint",
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh")
                load_btn = gr.Button("Load Checkpoint")
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=4,
                )
                with gr.Row():
                    max_tokens = gr.Slider(1, 1024, value=100, label="Max new tokens")
                    temperature = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                with gr.Row():
                    top_k = gr.Slider(0, 200, value=50, step=1, label="Top-K (0=off)")
                    top_p = gr.Slider(0.0, 1.0, value=0.9, label="Top-P")
                generate_btn = gr.Button("Generate", variant="primary")
                output = gr.Textbox(label="Output", lines=8, interactive=False)

        ckpt_map = {c[0]: c[1] for c in ckpts}

        def on_load(choice_name):
            path = ckpt_map.get(choice_name)
            if not path:
                return "Invalid checkpoint"
            return session.load_checkpoint(path)

        def on_refresh():
            nonlocal ckpt_map
            new_ckpts = list_checkpoints(checkpoint_dir)
            new_choices = [c[0] for c in new_ckpts]
            ckpt_map = {c[0]: c[1] for c in new_ckpts}
            return gr.Dropdown(
                choices=new_choices,
                value=new_choices[0] if new_choices else None,
            )

        def on_generate(prompt_text, max_tok, temp, topk, topp):
            return session.generate(prompt_text, max_tok, temp, topk, topp)

        load_btn.click(on_load, inputs=[ckpt_dropdown], outputs=[status])
        refresh_btn.click(on_refresh, inputs=[], outputs=[ckpt_dropdown])
        generate_btn.click(
            on_generate,
            inputs=[prompt, max_tokens, temperature, top_k, top_p],
            outputs=[output],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="SLM inference GUI")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer.json")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    session = InferenceSession(tokenizer_path=args.tokenizer)
    app = build_interface(session, args.checkpoint_dir)

    print(f"Launching GUI on {args.host}:{args.port}")
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
