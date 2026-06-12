from unittest.mock import MagicMock

import pytest

from wallgen import config as cfg
from wallgen import loras


def test_parse_cli_loras():
    specs = loras.parse_cli_loras(["org/style:0.7", "loras/local.safetensors", "a/b:notnum"])
    assert specs[0].source == "org/style" and specs[0].scale == 0.7
    assert specs[1].source == "loras/local.safetensors" and specs[1].scale == 1.0
    # Non-numeric tail is treated as part of the source.
    assert specs[2].source == "a/b:notnum" and specs[2].scale == 1.0


def test_resolve_stack_combines(tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    extra = [loras.LoraSpec(source="org/extra", scale=0.5)]
    assert loras.resolve_stack(conf, "none", extra) == extra


def test_apply_loras_empty_is_noop():
    pipe = MagicMock()
    assert loras.apply_loras(pipe, []) == []
    pipe.load_lora_weights.assert_not_called()


def test_apply_loras_loads_and_sets_adapters():
    pipe = MagicMock()
    specs = [loras.LoraSpec(source="org/a", scale=0.8), loras.LoraSpec(source="org/b", scale=0.5)]
    names = loras.apply_loras(pipe, specs)
    assert len(names) == 2
    assert pipe.load_lora_weights.call_count == 2
    pipe.set_adapters.assert_called_once()
    _, kwargs = pipe.set_adapters.call_args
    assert kwargs["adapter_weights"] == [0.8, 0.5]


def test_apply_loras_skips_failures():
    pipe = MagicMock()
    pipe.load_lora_weights.side_effect = [RuntimeError("boom"), None]
    specs = [loras.LoraSpec(source="bad"), loras.LoraSpec(source="good", scale=0.9)]
    applied = loras.apply_loras(pipe, specs)
    assert len(applied) == 1
    assert applied[0].source == "good"


def test_apply_loras_all_fail_raises():
    pipe = MagicMock()
    pipe.load_lora_weights.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        loras.apply_loras(pipe, [loras.LoraSpec(source="bad")])


def test_lora_metadata():
    specs = [loras.LoraSpec(source="org/a", scale=0.8)]
    assert loras.lora_metadata(specs) == [{"source": "org/a", "scale": 0.8}]


def test_list_local_loras(tmp_path):
    (tmp_path / "x.safetensors").write_bytes(b"x")
    (tmp_path / "y.txt").write_text("ignore")
    assert loras.list_local_loras(tmp_path) == ["x.safetensors"]
    assert loras.list_local_loras(tmp_path / "missing") == []
