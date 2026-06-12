# WallGen — Multi-Model Wallpaper Generator

A lean, self-hosted text-to-image pipeline for high-resolution desktop wallpapers. It picks the **most optimized model + backend for your hardware** automatically — MLX (`mflux`) on Apple Silicon, HuggingFace Diffusers + GGUF on CUDA/CPU — quantizes both the transformer *and* the text encoder for a small footprint, and generates with a wallpaper-aware prompt directive.

> 📖 Live docs: https://arslankazmi.github.io/wallgen/

## What it does

- **Best model per host, automatically.** A priority registry + memory probe pick the best model that fits: `z-image-turbo` (default) → `FLUX.2-klein-4B` → `sd-turbo` (floor). VRAM and system RAM are considered separately on CUDA.
- **Optimal backend per OS.** Apple Silicon runs MLX (`mflux`) for zero-copy unified-memory speed; CUDA/CPU run Diffusers with GGUF quantization and the right dtype (bf16 on Ampere+, fp16 on Turing, fp32 on CPU) and offload tier.
- **Tiny footprint.** Transformer *and* text encoder are quantized — e.g. z-image-turbo 4-bit on Apple Silicon is a **~6GB** download and runs under 8GB.
- **Lean output.** No upscaler: generate at the model's native size (≤1080p) and **stretch-resize** to your screen. The prompt directive tells the model the image will be stretched, so it composes accordingly.
- **Deployable prompt control.** A configurable wallpaper directive plus a prompt **lock** (`off`/`template`/`locked`) — the daily scheduler runs locked so it can't be hijacked.
- **LoRA-ready.** Stack third-party/local LoRAs at inference (both backends); train your own via the finetune workflow.
- **Private by default.** Gradio + HuggingFace telemetry disabled; UI binds to localhost.

## Hardware matrix (auto-selected)

| Host | Backend | Model | dtype | quant | placement |
|------|---------|-------|-------|-------|-----------|
| Apple Silicon (M-series), 16GB | MLX | z-image-turbo 4-bit | — | 4-bit | resident |
| RTX 3060 Ti 8GB / 32GB RAM | Diffusers CUDA | z-image-turbo | bf16 | Q4/Q5 | model-offload |
| GTX 1660 Ti 6GB / 16GB RAM | Diffusers CUDA | FLUX.2-klein-4B | fp16 | Q4 | model/seq offload |
| i3, no GPU, 8GB | Diffusers CPU | FLUX.2-klein-4B | fp32 | Q3/Q4 | sequential (slow) |
| i3, no GPU, 4GB (floor) | Diffusers CPU | sd-turbo | fp32 | — | sequential |

## Models

| Model | Maker / license | Why |
|-------|-----------------|-----|
| **z-image-turbo** (default) | Tongyi-MAI · Apache-2.0 | Best quality-per-byte; beats FLUX.1-schnell, rivals FLUX.2-dev; 9 steps |
| **FLUX.2-klein-4B** | Black Forest Labs | FLUX flagship; 4 steps; smaller VRAM need |
| **sd-turbo** | Stability AI | Tiny floor for 4GB / no-GPU hosts |

All are guidance-distilled (`guidance_scale=0`).

## Quick start

```bash
# Apple Silicon (MLX fast path)
uv sync --extra mlx --extra ui --extra lora

# NVIDIA GPU (Windows/Linux)
uv sync --no-group cpu --group cuda --extra ui

# No GPU (Windows/Linux) — lean CPU torch
uv sync

# Generate (model + backend auto-selected; output stretched to config target)
uv run python -m wallgen generate "a serene misty mountain at dawn, ultrawide"

# Batch, UI, config inspection
uv run python -m wallgen batch prompts/wallpapers.txt
uv run python -m wallgen ui            # localhost Gradio, no telemetry
uv run python -m wallgen list-models
```

## Output sizing

Configured in `config.yaml`:

```yaml
output:
  gen_size: [1024, 576]      # model-native generation (<=1080p)
  target_size: [1920, 1080]  # final wallpaper; image is stretched to this
  stretch: true
```

Generation and target share an aspect ratio by default, so the stretch is a clean scale.

## Prompt control (for the deployed pipeline)

```yaml
prompt:
  wallpaper_directive: "high-quality desktop wallpaper, full-bleed ... the image will be stretched to fill the screen ..."
  lock:
    mode: off        # off (raw) | template (subject + directive) | locked (only approved prompts)
    locked_prompts: prompts/wallpapers.txt
```

The **daily scheduler** always runs locked: it rotates through `prompts/wallpapers.txt`, composes the directive, generates at the prod profile, and sets your desktop. Install via `wallgen/scheduler/com.arslankazmi.wallgen.daily.plist` (macOS launchd) or `wallgen.cron` (Linux).

## LoRAs

```bash
uv run python -m wallgen generate "neon city" --lora some-org/style:0.8 --lora loras/mine.safetensors:0.6
```

Drop local `.safetensors` into `loras/`. Requesting a LoRA on the Diffusers path loads an unquantized base (fusing is unreliable on quantized weights). Train your own: `uv run python -m wallgen train --dataset <dir>`.

## Dependency footprint

torch lives in conflicting `cpu`/`cuda` dependency-groups so the multi-GB CUDA stack is never installed unless requested. Heavy features (`ui`, `lora`, `train`, `mlx`) are opt-in extras. No upscaler dependency.

## Development

```bash
uv run pytest        # pure logic + placement/selection tests; no GPU needed
```

## License & credits

Apache-2.0. Built on [Diffusers](https://github.com/huggingface/diffusers), [mflux](https://github.com/filipstrand/mflux), Tongyi-MAI [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), Black Forest Labs [FLUX](https://huggingface.co/black-forest-labs), Stability AI, and GGUF quants by [unsloth](https://huggingface.co/unsloth)/[city96](https://huggingface.co/city96). See [ASSET_CREDITS.md](ASSET_CREDITS.md).
