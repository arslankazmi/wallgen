"""LoRA resolution and application.

Supports a *stack* of adapters, each from an HF repo id or a local
``.safetensors`` file, with independent scales. Adapters are applied **without
fusing** because fusing is unreliable on quantized (GGUF) bases — the caller is
expected to have loaded a bf16 base when a stack is requested (see
``models.load_pipeline(use_loras=True)``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config, LoraSpec

log = logging.getLogger("fluxwall.loras")


def parse_cli_loras(items: list[str] | None) -> list[LoraSpec]:
    """Parse repeatable ``--lora source:scale`` CLI args into specs.

    ``source`` may itself contain colons only if no scale is given; a trailing
    ``:<float>`` is interpreted as the scale.
    """
    specs: list[LoraSpec] = []
    for raw in items or []:
        source, scale = raw, 1.0
        if ":" in raw:
            head, _, tail = raw.rpartition(":")
            try:
                scale = float(tail)
                source = head
            except ValueError:
                source, scale = raw, 1.0
        specs.append(LoraSpec(source=source, scale=scale))
    return specs


def resolve_stack(
    config: Config,
    stack_name: str | None,
    extra: list[LoraSpec] | None = None,
) -> list[LoraSpec]:
    """Combine a named config stack with any ad-hoc CLI specs."""
    specs = list(config.lora_stack(stack_name))
    if extra:
        specs.extend(extra)
    return specs


def _adapter_name(spec: LoraSpec, index: int) -> str:
    stem = Path(spec.source).stem.replace(".", "_").replace("/", "_")
    return f"{stem}_{index}" or f"lora_{index}"


def apply_loras(pipe, specs: list[LoraSpec]) -> list[LoraSpec]:
    """Load + activate a LoRA stack on a pipeline. No-op for an empty stack.

    Returns the list of specs that were *actually applied* (so callers record
    truth in the sidecar, not merely what was requested). Raises RuntimeError if
    a non-empty stack was requested but every adapter failed to load.
    """
    if not specs:
        return []
    names: list[str] = []
    scales: list[float] = []
    applied: list[LoraSpec] = []
    for i, spec in enumerate(specs):
        name = _adapter_name(spec, i)
        path = Path(spec.source)
        try:
            if path.exists():
                pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=name)
            else:
                pipe.load_lora_weights(spec.source, adapter_name=name)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load LoRA %s (%s); skipping", spec.source, exc)
            continue
        names.append(name)
        scales.append(spec.scale)
        applied.append(spec)
    if not applied:
        raise RuntimeError(
            f"All {len(specs)} requested LoRA adapter(s) failed to load; aborting "
            "so the output isn't silently un-styled."
        )
    pipe.set_adapters(names, adapter_weights=scales)
    log.info("Applied %d/%d LoRA adapter(s): %s", len(applied), len(specs), list(zip(names, scales)))
    return applied


def lora_metadata(specs: list[LoraSpec]) -> list[dict]:
    """Serialisable record of the LoRA stack for the image sidecar."""
    return [{"source": s.source, "scale": s.scale} for s in specs]


def list_local_loras(loras_dir: str | Path = "loras") -> list[str]:
    """Filenames of local ``.safetensors`` LoRAs available for drop-in use."""
    d = Path(loras_dir)
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("*.safetensors"))
