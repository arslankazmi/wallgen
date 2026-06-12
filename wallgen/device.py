"""Hardware capability probe, backend selection, and placement resolution.

All decision logic here is pure (string/number in, dataclass out) so it can be
unit-tested for every target host without a GPU. Heavy imports (torch, psutil)
are lazy.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

from .config import ModelSpec

# GGUF quant tiers, best first. Chosen by the device's binding memory (GiB).
_QUANT_TIERS: tuple[tuple[float, str], ...] = (
    (16.0, "Q8_0"),
    (12.0, "Q6_K"),
    (8.0, "Q5_K_M"),
    (6.0, "Q4_K_M"),
    (0.0, "Q3_K_M"),
)
# Below this device memory (GiB) the heavy text encoders can't stay resident.
_OFFLOAD_MEMORY_GB = 24.0


@dataclass(frozen=True)
class Capabilities:
    system: str  # "Darwin" | "Linux" | "Windows"
    arch: str  # "arm64" | "x86_64" | ...
    has_cuda: bool
    vram_gb: float
    cuda_cc: tuple[int, int]  # (major, minor); (0, 0) if no CUDA
    has_mps: bool
    ram_gb: float
    mlx_ok: bool

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.arch == "arm64"


@dataclass(frozen=True)
class Placement:
    device: str  # "cuda" | "mps" | "cpu"
    dtype: str  # torch dtype name
    quant: str  # GGUF tier, e.g. "Q4_K_M"
    offload: str  # "none" | "model" | "sequential"


# --- capability probe --------------------------------------------------------
def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    """True iff mflux + mlx import (only possible on macOS arm64 >= 14.0)."""
    try:
        import mlx.core  # noqa: F401
        import mflux  # noqa: F401

        return True
    except Exception:
        return False


def _system_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return 16.0


def probe_capabilities() -> Capabilities:
    """Best-effort hardware probe. Falls back to conservative values."""
    has_cuda = has_mps = False
    vram_gb = 0.0
    cuda_cc = (0, 0)
    try:
        import torch

        if torch.cuda.is_available():
            has_cuda = True
            cuda_cc = torch.cuda.get_device_capability(0)
            try:
                _free, total = torch.cuda.mem_get_info(0)
                vram_gb = total / (1024**3)
            except Exception:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            has_mps = True
    except Exception:
        pass
    return Capabilities(
        system=platform.system(),
        arch=platform.machine(),
        has_cuda=has_cuda,
        vram_gb=round(vram_gb, 1),
        cuda_cc=tuple(cuda_cc),
        has_mps=has_mps,
        ram_gb=round(_system_ram_gb(), 1),
        mlx_ok=is_apple_silicon() and mlx_available(),
    )


# --- selection logic (pure) --------------------------------------------------
def select_backend(caps: Capabilities, spec: ModelSpec) -> str:
    """'mlx' on Apple Silicon when mflux is installed and the model has an mlx
    source; otherwise 'diffusers'."""
    if caps.mlx_ok and spec.mlx:
        return "mlx"
    return "diffusers"


def _quant_for(memory_gb: float) -> str:
    for threshold, quant in _QUANT_TIERS:
        if memory_gb >= threshold:
            return quant
    return _QUANT_TIERS[-1][1]


def resolve_placement(caps: Capabilities, spec: ModelSpec, quant_override: str | None = None) -> Placement:
    """Decide device, dtype, quant, and offload mode for a model on this host."""
    if caps.has_cuda:
        device = "cuda"
        dtype = "bfloat16" if caps.cuda_cc[0] >= 8 else "float16"  # bf16 only Ampere+
        budget = caps.vram_gb
        quant = quant_override if (quant_override and quant_override != "auto") else _quant_for(budget)
        # VRAM is separate from RAM, so offload genuinely frees VRAM.
        if caps.vram_gb >= spec.min_memory_gb:
            offload = "none"  # whole model fits VRAM
        elif caps.vram_gb >= spec.min_vram_gb:
            offload = "model"  # transformer fits; encoder/VAE stream from RAM
        else:
            offload = "sequential"
    elif caps.has_mps:
        device = "mps"
        dtype = "bfloat16"
        # Unified RAM is shared with the OS — reserve headroom before picking quant.
        quant = quant_override if (quant_override and quant_override != "auto") else _quant_for(max(0.0, caps.ram_gb - 4))
        offload = "none"  # unified memory: offload only duplicates weights
    else:
        device = "cpu"
        dtype = "float32"
        quant = quant_override if (quant_override and quant_override != "auto") else _quant_for(max(0.0, caps.ram_gb - 4))
        offload = "sequential" if spec.min_memory_gb > 6 else "none"
    return Placement(device=device, dtype=dtype, quant=quant, offload=offload)


# --- torch helpers (lazy) ----------------------------------------------------
def torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def empty_cache(device: str) -> None:
    try:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass
