"""LoRA finetuning entry point.

This intentionally does NOT reimplement a diffusion trainer. It validates the
dataset, resolves a training config template, and hands off to the established
finetuning workflow (the ``ak:ml-finetune`` skill — SimpleTuner / kohya for FLUX
LoRA). The resulting ``.safetensors`` is written into ``loras/`` where the
inference path (``wallgen.loras``) picks it up automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import dataset as ds

log = logging.getLogger("wallgen.train")

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def resolve_config(config_name: str) -> Path:
    for ext in (".toml", ".yaml", ".yml"):
        candidate = CONFIG_DIR / f"{config_name}{ext}"
        if candidate.exists():
            return candidate
    available = sorted(p.name for p in CONFIG_DIR.glob("*"))
    raise FileNotFoundError(f"Training config {config_name!r} not found. Available: {available}")


def run(dataset_dir: str, config_name: str = "flux_lora_default", output_dir: str = "loras") -> dict:
    """Validate inputs and emit the finetune hand-off plan.

    Returns a dict describing the resolved run. Actual training is launched by
    the ak:ml-finetune workflow using ``config_path`` + ``dataset_dir``.
    """
    examples = ds.load_dataset(dataset_dir)
    config_path = resolve_config(config_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plan = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "num_examples": len(examples),
        "config_path": str(config_path),
        "output_dir": str(Path(output_dir).resolve()),
    }
    log.info("LoRA finetune ready: %d examples, config=%s", len(examples), config_path.name)
    log.info(
        "Hand off to ak:ml-finetune (FLUX LoRA via SimpleTuner/kohya):\n"
        "  dataset=%s  config=%s  output=%s",
        plan["dataset_dir"], plan["config_path"], plan["output_dir"],
    )
    return plan
