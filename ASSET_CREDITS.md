# Asset & Dependency Credits

This project stands on the work of others. Thank you to:

## Models
- **Tongyi-MAI — Z-Image-Turbo** (default model) — efficient S3-DiT text-to-image. Apache-2.0.
  - `Tongyi-MAI/Z-Image-Turbo` · https://github.com/Tongyi-MAI/Z-Image
- **Black Forest Labs — FLUX.2-klein-4B** — FLUX flagship. Apache-2.0.
  - `black-forest-labs/FLUX.2-klein-4B`
- **Stability AI — sd-turbo** — tiny floor model for low-memory hosts.
  - `stabilityai/sd-turbo`

## Backends & quantizations
- **mflux** (Filip Strand) — FLUX/Z-Image on Apple MLX. MIT.
  - https://github.com/filipstrand/mflux · pre-quant repos `filipstrand/Z-Image-Turbo-mflux-4bit`, `Runpod/FLUX.2-klein-4B-mflux-4bit`
- **unsloth** — GGUF quantizations (transformer + Qwen3 text encoder).
  - `unsloth/Z-Image-Turbo-GGUF`, `unsloth/FLUX.2-klein-4B-GGUF`, `unsloth/Qwen3-4B-GGUF`
- **city96** — GGUF quantizations & tooling.

## Libraries
- **HuggingFace Diffusers / Transformers / Hub** — pipelines, model loading, GGUF support, fast downloads (`hf_transfer`).
- **MLX** (Apple) — Apple Silicon array framework powering the fast path.
- **PyTorch**, **Typer**, **Gradio**, **PEFT**, **Pillow** — runtime, CLI, UI, LoRA, image I/O.

## Tooling
- LoRA finetuning delegates to **SimpleTuner** / **kohya-ss** via the project's finetune workflow.

All trademarks and models are the property of their respective owners and used under their respective licenses.
