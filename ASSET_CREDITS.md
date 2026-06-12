# Asset & Dependency Credits

This project stands on the work of others. Thank you to:

## Models
- **Black Forest Labs — FLUX.1 & FLUX.2** — the text-to-image models powering this pipeline.
  - `black-forest-labs/FLUX.1-schnell` (Apache-2.0)
  - `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0)
  - https://huggingface.co/black-forest-labs
- **Stability AI — SDXL-Turbo** — fast fallback model.
  - `stabilityai/sdxl-turbo`

## Quantizations
- **city96** — GGUF quantizations of FLUX.1 and the T5 text encoder.
  - `city96/FLUX.1-schnell-gguf`, `city96/t5-v1_1-xxl-encoder-gguf`
  - https://huggingface.co/city96
- **Unsloth** — GGUF quantizations of FLUX.2-klein.
  - `unsloth/FLUX.2-klein-4B-GGUF`
  - https://huggingface.co/unsloth

## Libraries
- **HuggingFace Diffusers / Transformers / Hub** — pipeline, model loading, GGUF support.
- **Real-ESRGAN** (Xintao Wang et al.) via `realesrgan-ncnn-py` — high-resolution upscaling.
  - https://github.com/xinntao/Real-ESRGAN
- **PyTorch**, **Typer**, **Gradio**, **PEFT** — runtime, CLI, UI, and LoRA support.

## Tooling
- LoRA finetuning delegates to **SimpleTuner** / **kohya-ss** via the project's finetune workflow.

All trademarks and models are the property of their respective owners and used under their respective licenses.
