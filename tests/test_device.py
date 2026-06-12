from fluxwall import device


def test_select_quant_tiers():
    assert device.select_quant(64) == "Q8_0"
    assert device.select_quant(32) == "Q8_0"
    assert device.select_quant(24) == "Q6_K"
    assert device.select_quant(18) == "Q5_K_M"
    assert device.select_quant(16) == "Q4_K_M"
    assert device.select_quant(8) == "Q4_K_M"


def test_dtype_for():
    assert device.dtype_for("cuda") == "bfloat16"
    assert device.dtype_for("mps") == "bfloat16"
    assert device.dtype_for("cpu") == "float32"


def test_resolve_profile_uses_overrides(monkeypatch):
    monkeypatch.setattr(device, "detect_device", lambda: "mps")
    monkeypatch.setattr(device, "available_memory_gb", lambda d=None: 24.0)

    auto = device.resolve_profile("auto")
    assert auto.device == "mps"
    assert auto.dtype == "bfloat16"
    assert auto.quant == "Q6_K"
    assert auto.supports_bf16 is True

    fixed = device.resolve_profile("Q4_K_M")
    assert fixed.quant == "Q4_K_M"


def test_resolve_profile_cpu(monkeypatch):
    monkeypatch.setattr(device, "detect_device", lambda: "cpu")
    monkeypatch.setattr(device, "available_memory_gb", lambda d=None: 8.0)
    prof = device.resolve_profile(None)
    assert prof.device == "cpu"
    assert prof.dtype == "float32"
    assert prof.quant == "Q4_K_M"
    assert prof.supports_bf16 is False
