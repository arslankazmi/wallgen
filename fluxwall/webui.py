"""Local Gradio web UI for interactive wallpaper generation.

Privacy: Gradio analytics are disabled here (and globally via env vars set in
``fluxwall/__init__.py``). The server binds to localhost and never creates a
public share link.
"""

from __future__ import annotations

import logging

from . import config as cfg
from . import loras as lora_mod

log = logging.getLogger("fluxwall.webui")


def _build(config_path: str | None = None):
    import gradio as gr

    conf = cfg.load_config(config_path)
    profile_names = list(conf.profiles)
    resolution_names = list(conf.resolutions)
    stack_names = list(conf.lora_stacks)
    default_profile = conf.default_profile

    def _generate(prompt, profile, resolution, stack, scale, seed):
        from . import pipeline

        if not prompt or not prompt.strip():
            raise gr.Error("Please enter a prompt.")
        extra = []
        # A single optional scale slider lets users dial the chosen stack up/down.
        base_specs = lora_mod.resolve_stack(conf, None if stack == "none" else stack)
        if base_specs and scale is not None:
            base_specs = [lora_mod.LoraSpec(source=s.source, scale=float(scale)) for s in base_specs]
        records = pipeline.generate(
            [prompt.strip()],
            profile_name=profile,
            resolution_name=resolution,
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

    with gr.Blocks(title="Flux Wallpaper Generator", analytics_enabled=False) as demo:
        gr.Markdown("# 🖼️ Flux Wallpaper Generator\nSelf-hosted FLUX text-to-image — no data leaves this machine.")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a serene misty mountain range at dawn, ultrawide")
                with gr.Row():
                    profile = gr.Dropdown(profile_names, value=default_profile, label="Profile")
                    resolution = gr.Dropdown(resolution_names, value="hd", label="Resolution")
                with gr.Row():
                    stack = gr.Dropdown(stack_names, value="none", label="LoRA stack")
                    scale = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="LoRA scale")
                seed = gr.Number(label="Seed (blank = random)", precision=0)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=3):
                output = gr.Gallery(label="Result", columns=1, height=420)
        gr.Markdown("### Recent wallpapers")
        gallery = gr.Gallery(value=_gallery_items, label="output/", columns=4, height=300)

        btn.click(_generate, [prompt, profile, resolution, stack, scale, seed], output).then(
            _gallery_items, None, gallery
        )
    return demo


def launch(host: str = "127.0.0.1", port: int = 7864, config_path: str | None = None) -> None:
    demo = _build(config_path)
    demo.launch(server_name=host, server_port=port, share=False, analytics_enabled=False)
