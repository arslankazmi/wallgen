import pytest

from fluxwall import config as cfg


def test_load_defaults_without_file(tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    assert conf.default_profile == "dev"
    assert "dev" in conf.profiles and "prod" in conf.profiles
    assert conf.profile().model == "flux2-klein-4b"
    assert conf.model("flux1-schnell").kind == "flux"
    assert conf.model("flux2-klein-4b").fallback == "sdxl-lightning"
    assert conf.resolution("hd").gen == (1024, 576)
    assert conf.resolution("hd").target == (1920, 1080)
    assert conf.lora_stack("none") == []


def test_yaml_overrides_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
default_profile: prod
profiles:
  prod: { model: flux1-schnell, steps: 6, guidance: 1.0 }
loras:
  cinematic:
    - { source: org/lora-a, scale: 0.7 }
    - org/lora-b
""".strip()
    )
    conf = cfg.load_config(p)
    assert conf.default_profile == "prod"
    assert conf.profile().steps == 6
    assert conf.profile().guidance == 1.0
    # Unspecified models still present from defaults (deep merge).
    assert "flux2-klein-4b" in conf.models
    stack = conf.lora_stack("cinematic")
    assert stack[0].source == "org/lora-a" and stack[0].scale == 0.7
    assert stack[1].source == "org/lora-b" and stack[1].scale == 1.0


def test_fit_model_by_memory(tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    # 16 GB host asking for the heavy prod tier -> downgrades to klein (fits).
    assert conf.fit_model("flux1-schnell", 16.0) == "flux2-klein-4b"
    # Plenty of memory -> keeps the requested model.
    assert conf.fit_model("flux1-schnell", 64.0) == "flux1-schnell"
    # Requested model already fits -> unchanged.
    assert conf.fit_model("flux2-klein-4b", 16.0) == "flux2-klein-4b"
    # Tiny host -> smallest available (sdxl, min 8) since no FLUX fits.
    assert conf.fit_model("flux2-klein-4b", 8.0) == "sdxl-lightning"


def test_unknown_lookups_raise(tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    with pytest.raises(KeyError):
        conf.profile("ghost")
    with pytest.raises(KeyError):
        conf.model("ghost")
    with pytest.raises(KeyError):
        conf.resolution("ghost")
    with pytest.raises(KeyError):
        conf.lora_stack("ghost")
