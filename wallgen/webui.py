"""Local Gradio web UI for interactive wallpaper generation.

Privacy: Gradio analytics are disabled here (and globally via env vars set in
``wallgen/__init__.py``). The server binds to localhost and never creates a
public share link.
"""

from __future__ import annotations

import logging

from . import config as cfg
from . import loras as lora_mod
from . import prompts as prompt_mod

log = logging.getLogger("wallgen.webui")


def _build(config_path: str | None = None):
    import gradio as gr

    conf = cfg.load_config(config_path)
    profile_names = list(conf.profiles)
    stack_names = list(conf.lora_stacks)
    default_profile = conf.default_profile
    tw, th = conf.target_size

    def _generate(prompt, profile, stack, scale, seed):
        from . import pipeline

        if not prompt or not prompt.strip():
            raise gr.Error("Please enter a prompt.")
        # Honour the configured prompt lock + wallpaper directive.
        composed, _ = prompt_mod.build_prompts(conf, [prompt.strip()], single=True)
        base_specs = lora_mod.resolve_stack(conf, None if stack == "none" else stack)
        if base_specs and scale is not None:
            base_specs = [lora_mod.LoraSpec(source=s.source, scale=float(scale)) for s in base_specs]
        records = pipeline.generate(
            composed,
            profile_name=profile,
            extra_loras=base_specs,
            seed=int(seed) if seed not in (None, "") else None,
            config=conf,
        )
        return [r.image_path for r in records]

    def _gallery_items():
        out_dir = conf.output_dir
        if not out_dir.exists():
            return []
        return sorted((str(p) for p in out_dir.rglob("*.png")), reverse=True)[:60]

    with gr.Blocks(title="WallGen", analytics_enabled=False) as demo:
        gr.Markdown(
            f"# 🖼️ WallGen\n"
            f"Self-hosted text-to-image — no data leaves this machine. "
            f"Output stretched to **{tw}×{th}**."
        )
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a serene misty mountain range at dawn, ultrawide")
                with gr.Row():
                    profile = gr.Dropdown(profile_names, value=default_profile, label="Profile")
                    stack = gr.Dropdown(stack_names, value="none", label="LoRA stack")
                scale = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="LoRA scale")
                seed = gr.Number(label="Seed (blank = random)", precision=0)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=3):
                output = gr.Gallery(label="Result", columns=1, height=420)
        gr.Markdown("### Recent wallpapers")
        gallery = gr.Gallery(value=_gallery_items, label="output/", columns=4, height=300)

        btn.click(_generate, [prompt, profile, stack, scale, seed], output).then(
            _gallery_items, None, gallery
        )
    return demo


def launch(host: str = "127.0.0.1", port: int = 7864, config_path: str | None = None) -> None:
    demo = _build(config_path)
    demo.launch(server_name=host, server_port=port, share=False, analytics_enabled=False)
