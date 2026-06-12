import datetime as dt

import pytest

from wallgen import prompts


def test_slugify():
    assert prompts.slugify("A serene Misty Mountain!") == "a-serene-misty-mountain"
    assert prompts.slugify("   ") == "wallpaper"
    assert len(prompts.slugify("x" * 200)) <= 48


def test_read_prompt_file(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("# comment\n\nfirst prompt\nsecond prompt\n")
    assert prompts.read_prompt_file(p) == ["first prompt", "second prompt"]


def test_read_prompt_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        prompts.read_prompt_file(tmp_path / "missing.txt")


def test_resolve_prompts_combines(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("from file\n")
    assert prompts.resolve_prompts("inline", p) == ["inline", "from file"]
    with pytest.raises(ValueError):
        prompts.resolve_prompts(None, None)


def test_rotate_daily_is_deterministic():
    items = ["a", "b", "c"]
    d = dt.date(2026, 6, 12)
    first = prompts.rotate_daily(items, d)
    assert first == prompts.rotate_daily(items, d)
    # Consecutive days advance by one.
    nxt = prompts.rotate_daily(items, d + dt.timedelta(days=1))
    assert items.index(nxt) == (items.index(first) + 1) % len(items)
    with pytest.raises(ValueError):
        prompts.rotate_daily([])


def _config_with_lock(tmp_path, mode):
    from wallgen import config as cfg

    locked = tmp_path / "locked.txt"
    locked.write_text("alpha vista\nbeta vista\n")
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
prompt:
  wallpaper_directive: "DIRECTIVE"
  lock: {{ mode: {mode}, template: "{{prompt}} | {{wallpaper_directive}}", locked_prompts: {locked} }}
""".strip()
    )
    return cfg.load_config(p)


def test_build_prompts_off(tmp_path):
    conf = _config_with_lock(tmp_path, "off")
    out, overridden = prompts.build_prompts(conf, ["my subject"])
    assert out == ["my subject"] and overridden is False


def test_build_prompts_template(tmp_path):
    conf = _config_with_lock(tmp_path, "template")
    out, overridden = prompts.build_prompts(conf, ["my subject"])
    assert out == ["my subject | DIRECTIVE"] and overridden is False


def test_build_prompts_locked_ignores_user(tmp_path):
    conf = _config_with_lock(tmp_path, "locked")
    out, overridden = prompts.build_prompts(conf, ["hijack attempt"], single=True)
    assert overridden is True
    assert len(out) == 1
    assert "hijack" not in out[0]
    assert out[0].endswith("| DIRECTIVE")  # composed from a locked prompt
    # Batch (single=False) -> all locked prompts composed.
    out_all, _ = prompts.build_prompts(conf, [], single=False)
    assert len(out_all) == 2
