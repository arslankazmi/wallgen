import datetime as dt

import pytest

from fluxwall import prompts


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
