"""Capability-driven placement + backend selection — the cross-hardware logic,
pure and host-independent."""

from wallgen import device
from wallgen.config import ModelSpec
from wallgen.device import Capabilities

ZIMAGE = ModelSpec(name="z-image-turbo", kind="zimage", min_memory_gb=10, min_vram_gb=7, mlx={"quant": 4})
KLEIN = ModelSpec(name="flux2-klein-4b", kind="flux2", min_memory_gb=8, min_vram_gb=4, mlx={"quant": 4})


def _caps(**kw):
    base = dict(system="Linux", arch="x86_64", has_cuda=False, vram_gb=0.0,
               cuda_cc=(0, 0), has_mps=False, ram_gb=16.0, mlx_ok=False)
    base.update(kw)
    return Capabilities(**base)


def test_dtype_rule():
    # Ampere (cc>=8) -> bf16; Turing (7.5) -> fp16.
    p_ampere = device.resolve_placement(_caps(has_cuda=True, vram_gb=8, cuda_cc=(8, 6)), ZIMAGE)
    assert p_ampere.dtype == "bfloat16" and p_ampere.device == "cuda"
    p_turing = device.resolve_placement(_caps(has_cuda=True, vram_gb=6, cuda_cc=(7, 5)), KLEIN)
    assert p_turing.dtype == "float16"
    p_mps = device.resolve_placement(_caps(has_mps=True, ram_gb=16), KLEIN)
    assert p_mps.dtype == "bfloat16" and p_mps.device == "mps" and p_mps.offload == "none"
    p_cpu = device.resolve_placement(_caps(ram_gb=8), KLEIN)
    assert p_cpu.dtype == "float32" and p_cpu.device == "cpu"


def test_cuda_offload_modes():
    # 8GB VRAM, klein (vram req 4, total 8): whole model fits VRAM -> none.
    p = device.resolve_placement(_caps(has_cuda=True, vram_gb=8, cuda_cc=(8, 6)), KLEIN)
    assert p.offload == "none"
    # 8GB VRAM, z-image (vram 7, total 10): transformer fits, total doesn't -> model offload.
    p = device.resolve_placement(_caps(has_cuda=True, vram_gb=8, cuda_cc=(8, 6)), ZIMAGE)
    assert p.offload == "model"
    # 6GB VRAM, z-image (vram req 7): doesn't even fit transformer -> sequential.
    p = device.resolve_placement(_caps(has_cuda=True, vram_gb=6, cuda_cc=(7, 5)), ZIMAGE)
    assert p.offload == "sequential"


def test_quant_by_memory():
    # MPS reserves ~4GB OS headroom: 16GB -> budget 12 -> Q6_K.
    assert device.resolve_placement(_caps(has_mps=True, ram_gb=16), ZIMAGE).quant == "Q6_K"
    # CUDA VRAM is raw (dedicated).
    assert device.resolve_placement(_caps(has_cuda=True, vram_gb=6, cuda_cc=(7, 5)), KLEIN).quant == "Q4_K_M"
    assert device.resolve_placement(_caps(has_cuda=True, vram_gb=24, cuda_cc=(8, 9)), ZIMAGE).quant == "Q8_0"


def test_select_backend():
    apple = _caps(system="Darwin", arch="arm64", has_mps=True, mlx_ok=True)
    assert device.select_backend(apple, ZIMAGE) == "mlx"
    # No mlx source -> diffusers even on Apple.
    assert device.select_backend(apple, ModelSpec(name="x", kind="sdxl")) == "diffusers"
    # Non-Apple -> diffusers.
    assert device.select_backend(_caps(has_cuda=True, cuda_cc=(8, 6)), ZIMAGE) == "diffusers"
