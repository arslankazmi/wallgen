import datetime as dt
import json

from wallgen import organize


def test_output_subdir_by_date(tmp_path):
    day = dt.date(2026, 6, 12)
    assert organize.output_subdir(tmp_path, "date", day) == tmp_path / "2026-06-12"
    assert organize.output_subdir(tmp_path, "flat") == tmp_path


def test_target_paths_are_unique(tmp_path):
    png1, side1 = organize.target_paths(tmp_path, "misty mountain")
    png1.write_bytes(b"x")
    side1.write_text("{}")
    png2, side2 = organize.target_paths(tmp_path, "misty mountain")
    assert png1 != png2
    assert png2.name.startswith("misty-mountain")
    assert side2.suffix == ".json"


def test_write_sidecar_roundtrip(tmp_path):
    rec = organize.GenerationRecord(
        prompt="a prompt, wallpaper directive",
        model="z-image-turbo",
        backend="mlx",
        profile="dev",
        device="mps",
        quant="4bit",
        steps=9,
        guidance=0.0,
        seed=42,
        gen_size=(1024, 576),
        target_size=(1920, 1080),
        stretched=True,
        loras=[{"source": "org/x", "scale": 0.8}],
        created_at=organize.now_iso(),
        image_path="out.png",
    )
    side = tmp_path / "rec.json"
    organize.write_sidecar(side, rec)
    data = json.loads(side.read_text())
    assert data["backend"] == "mlx"
    assert data["model"] == "z-image-turbo"
    assert data["stretched"] is True
    assert data["target_size"] == [1920, 1080]
    assert data["loras"][0]["scale"] == 0.8
