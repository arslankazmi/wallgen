"""Device, dtype, and quantization selection.

Auto-detects the best available backend (CUDA > MPS > CPU) and picks a sane
dtype and GGUF quantization level based on how much memory the device exposes.
All heavy imports (torch, psutil) are done lazily so this module — and the tests
that exercise its pure logic — load fast and without a GPU stack.
"""

from __future__ import annotations

from dataclasses import dataclass

# GGUF quant tiers, largest/best first. Chosen by available memory in GiB.
# (transformer + text encoder + VAE + OS headroom must fit in unified/VRAM.)
_QUANT_TIERS: tuple[tuple[float, str], ...] = (
    (32.0, "Q8_0"),
    (24.0, "Q6_K"),
    (18.0, "Q5_K_M"),
    (0.0, "Q4_K_M"),
)


@dataclass(frozen=True)
class DeviceProfile:
    """Resolved hardware profile for a generation run."""

    device: str  # "cuda" | "mps" | "cpu"
    dtype: str  # torch dtype name: "bfloat16" | "float32"
    total_memory_gb: float
    quant: str  # auto-selected GGUF tier, e.g. "Q5_K_M"

    @property
    def supports_bf16(self) -> bool:
        return self.dtype == "bfloat16"


def detect_device() -> str:
    """Return the best available torch backend: cuda > mps > cpu."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch always present at runtime
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for(device: str) -> str:
    """bf16 on GPU backends (wider exponent range, native on M2+/modern CUDA);
    fp32 on CPU where bf16 is slow/unsupported."""
    return "bfloat16" if device in ("cuda", "mps") else "float32"


def available_memory_gb(device: str | None = None) -> float:
    """Best-effort total memory in GiB for the active device.

    CUDA -> total VRAM; MPS/CPU -> total system (unified) RAM. Falls back to a
    conservative 16 GiB if probing fails.
    """
    device = device or detect_device()
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return total_bytes / (1024**3)
    except Exception:
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return 16.0


def select_quant(memory_gb: float) -> str:
    """Pick the richest GGUF quant tier that should fit in `memory_gb`."""
    for threshold, quant in _QUANT_TIERS:
        if memory_gb >= threshold:
            return quant
    return _QUANT_TIERS[-1][1]


def resolve_profile(quant_override: str | None = None) -> DeviceProfile:
    """Full hardware probe -> DeviceProfile. `quant_override` ("auto" or None
    triggers auto-selection; anything else is used verbatim)."""
    device = detect_device()
    dtype = dtype_for(device)
    mem = available_memory_gb(device)
    if quant_override and quant_override.lower() != "auto":
        quant = quant_override
    else:
        quant = select_quant(mem)
    return DeviceProfile(device=device, dtype=dtype, total_memory_gb=round(mem, 1), quant=quant)


def torch_dtype(name: str):
    """Map a dtype name to the actual torch dtype object (lazy import)."""
    import torch

    return getattr(torch, name)


def empty_cache(device: str) -> None:
    """Release cached allocator memory for the active device (best effort)."""
    try:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass
