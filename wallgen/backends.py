"""Generation backends and selection.

Two backends behind one tiny interface:

* ``MLXBackend`` — Apple Silicon via ``mflux`` (zero-copy unified memory). mflux
  owns quantization, the text encoder, LoRA, and sampling; this is a thin shim.
* ``DiffusersBackend`` — everything else (CUDA/CPU). One generic loader driven by
  a ``kind -> (PipelineClass, TransformerClass)`` table handles z-image / FLUX.2
  / SDXL, with GGUF transformer + GGUF text-encoder quantization and a 3-mode
  placement (resident / model-offload / sequential-offload).

``build_backend`` resolves a model to a ready backend, transparently falling back
to the model's configured ``fallback`` (cycle-guarded) when a backend or model
kind is unavailable on this host.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import device as dev
from . import loras as lora_mod
from .config import Config, LoraSpec, ModelSpec
from .device import Capabilities, Placement

log = logging.getLogger("wallgen.backends")


class ModelUnsupported(RuntimeError):
    """Raised when a backend can't load a model kind on this host."""


# --- shared GGUF helpers -----------------------------------------------------
def _pick_gguf_filename(repo: str, quant: str) -> str:
    """Filename of the GGUF in ``repo`` matching ``quant`` (smallest match)."""
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    matches = [f for f in files if quant.lower() in f.lower()] or files
    if not matches:
        raise ModelUnsupported(f"No GGUF file in {repo!r}")
    return sorted(matches, key=len)[0]


def _gguf_local_path(repo: str, quant: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, _pick_gguf_filename(repo, quant))


# =============================================================================
# MLX backend (Apple Silicon / mflux)
# =============================================================================
class MLXBackend:
    name = "mlx"

    def __init__(self, config: Config, spec: ModelSpec, placement: Placement, lora_specs: list[LoraSpec]):
        self.spec = spec
        self.model = spec.name
        self.device = "mps"
        mlxcfg = spec.mlx or {}
        self.quant = f"{mlxcfg.get('quant', 4)}bit"
        self.active_loras = lora_specs
        self._model = self._load(spec, mlxcfg, lora_specs)

    def _load(self, spec: ModelSpec, mlxcfg: dict, lora_specs: list[LoraSpec]):
        try:
            if spec.kind == "zimage":
                from mflux.models.z_image import ZImageTurbo as Model
            elif spec.kind == "flux2":
                from mflux.models.flux2 import Flux2Klein as Model
            else:
                raise ModelUnsupported(f"mflux has no path for kind {spec.kind!r}")
        except Exception as exc:  # noqa: BLE001 - mflux missing/incompatible
            raise ModelUnsupported(f"mflux unavailable for {spec.name}: {exc}") from exc

        # Use the pre-quantized repo (small download, encoder already quantized)
        # instead of letting mflux pull + quantize the full base.
        model_path = None
        repo = mlxcfg.get("repo")
        if repo:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(repo)
        lora_paths = [_resolve_lora_path(s) for s in lora_specs] if lora_specs else None
        lora_scales = [s.scale for s in lora_specs] if lora_specs else None

        kwargs = dict(lora_paths=lora_paths, lora_scales=lora_scales)
        if model_path:
            # Pre-quantized weights — do NOT re-quantize (the repo is already 4-bit).
            kwargs["model_path"] = model_path
        else:
            # Quantizing a full base model on the fly.
            kwargs["quantize"] = mlxcfg.get("quant", 4)
        return Model(**kwargs)

    def generate(self, prompt, width, height, steps, guidance, seed):
        return self._model.generate_image(
            seed=int(seed or 0),
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance=guidance if guidance else None,
        )


def _resolve_lora_path(spec: LoraSpec) -> str:
    """A local file path for a LoRA (download from HF if it's a repo id)."""
    p = Path(spec.source)
    if p.exists():
        return str(p)
    from huggingface_hub import hf_hub_download
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(spec.source) if f.endswith(".safetensors")]
    if not files:
        raise ModelUnsupported(f"No .safetensors in LoRA repo {spec.source!r}")
    return hf_hub_download(spec.source, sorted(files, key=len)[0])


# =============================================================================
# Diffusers backend (CUDA / CPU / non-Apple)
# =============================================================================
# kind -> (pipeline class name, transformer class name | None)
_DIFFUSERS_CLASSES = {
    "zimage": ("ZImagePipeline", "ZImageTransformer2DModel"),
    "flux2": ("DiffusionPipeline", "Flux2Transformer2DModel"),  # auto-resolves klein's pipeline
    "sdxl": ("AutoPipelineForText2Image", None),
}
_FLUX_KINDS = {"flux2"}  # accept max_sequence_length


class DiffusersBackend:
    name = "diffusers"

    def __init__(self, config: Config, spec: ModelSpec, placement: Placement, lora_specs: list[LoraSpec]):
        self.spec = spec
        self.model = spec.name
        self.placement = placement
        self.device = placement.device
        self.active_loras = lora_specs
        self._use_gguf = not lora_specs  # adapters attach to unquantized weights only
        self.quant = placement.quant if (self._use_gguf and spec.gguf) else "none"
        self._pipe = self._load(spec, placement)
        self.active_loras = lora_mod.apply_loras(self._pipe, lora_specs)

    def _resolve_classes(self, kind: str):
        import diffusers

        if kind not in _DIFFUSERS_CLASSES:
            raise ModelUnsupported(f"Unknown kind {kind!r}")
        pipe_name, tf_name = _DIFFUSERS_CLASSES[kind]
        try:
            PipelineClass = getattr(diffusers, pipe_name)
            TransformerClass = getattr(diffusers, tf_name) if tf_name else None
        except AttributeError as exc:
            raise ModelUnsupported(f"diffusers lacks {pipe_name}: {exc}") from exc
        return PipelineClass, TransformerClass

    def _load(self, spec: ModelSpec, placement: Placement):
        from .device import torch_dtype

        PipelineClass, TransformerClass = self._resolve_classes(spec.kind)
        dtype_name = placement.dtype
        # The small SD floor model NaNs (black images) in bf16 on MPS — use fp32.
        if spec.kind == "sdxl" and placement.device == "mps":
            dtype_name = "float32"
        dtype = torch_dtype(dtype_name)
        kwargs: dict = {"torch_dtype": dtype}

        # Quantized transformer (GGUF) — only when not applying LoRA.
        if self._use_gguf and spec.gguf and TransformerClass is not None:
            try:
                from diffusers import GGUFQuantizationConfig

                ckpt = _gguf_local_path(spec.gguf, placement.quant)
                kwargs["transformer"] = TransformerClass.from_single_file(
                    ckpt,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                    torch_dtype=dtype,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("GGUF transformer load failed (%s); using bf16 base", exc)

        # Quantized text encoder (GGUF Qwen3 for zimage/flux2).
        if self._use_gguf and spec.text_encoder_gguf:
            try:
                from transformers import AutoModelForCausalLM

                fname = _pick_gguf_filename(spec.text_encoder_gguf, placement.quant)
                kwargs["text_encoder"] = AutoModelForCausalLM.from_pretrained(
                    spec.text_encoder_gguf, gguf_file=fname, torch_dtype=dtype
                )
            except Exception as exc:  # noqa: BLE001 - qwen3 GGUF arch can be flaky
                log.warning("GGUF text encoder load failed (%s); using bf16 encoder", exc)

        try:
            pipe = PipelineClass.from_pretrained(spec.base, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnsupported(f"Could not load {spec.base}: {exc}") from exc
        return self._place(pipe, placement)

    @staticmethod
    def _place(pipe, placement: Placement):
        if placement.offload == "model":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:  # noqa: BLE001
                pipe.to(placement.device)
        elif placement.offload == "sequential":
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:  # noqa: BLE001
                pipe.to(placement.device)
        else:
            pipe.to(placement.device)
        # SD/SDXL VAEs produce NaNs (black images) in fp16/bf16 — decode in fp32.
        try:
            pipe.vae.config.force_upcast = True
        except Exception:  # noqa: BLE001
            pass
        for saver in ("enable_vae_tiling", "enable_attention_slicing"):
            try:
                getattr(pipe, saver)()
            except Exception:  # noqa: BLE001
                pass
        return pipe

    def generate(self, prompt, width, height, steps, guidance, seed):
        import torch

        gen = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        )
        if self.spec.kind in _FLUX_KINDS:
            kwargs["max_sequence_length"] = 256
        return self._pipe(**kwargs).images[0]


# =============================================================================
# Selection + construction
# =============================================================================
def build_backend(
    config: Config,
    model_name: str,
    caps: Capabilities,
    lora_specs: list[LoraSpec] | None = None,
    quant_override: str | None = None,
    _visited: frozenset[str] | None = None,
):
    """Resolve ``model_name`` to a ready backend for this host.

    Picks MLX on Apple Silicon (falling back to diffusers for the same model if
    mflux can't load it), and on any ``ModelUnsupported`` degrades to the model's
    configured ``fallback`` (cycle-guarded).
    """
    visited = (_visited or frozenset()) | {model_name}
    spec = config.model(model_name)
    lora_specs = lora_specs or []
    placement = dev.resolve_placement(caps, spec, quant_override)
    kind_backend = dev.select_backend(caps, spec)

    try:
        if kind_backend == "mlx":
            try:
                return MLXBackend(config, spec, placement, lora_specs)
            except ModelUnsupported as exc:
                log.warning("MLX failed for %s (%s); trying diffusers", model_name, exc)
        return DiffusersBackend(config, spec, placement, lora_specs)
    except ModelUnsupported as exc:
        if spec.fallback and spec.fallback in config.models and spec.fallback not in visited:
            log.warning("%s unsupported (%s); falling back to %s", model_name, exc, spec.fallback)
            return build_backend(config, spec.fallback, caps, lora_specs, quant_override, visited)
        raise
