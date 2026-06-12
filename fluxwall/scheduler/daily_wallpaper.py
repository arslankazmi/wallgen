"""The once-a-day off-hours job: pick today's prompt, generate at the prod tier,
optionally set it as the desktop wallpaper. Invoked by launchd (macOS) or cron
(Linux) — see the templates alongside this module.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from .. import config as cfg
from .. import loras as lora_mod
from .. import prompts as prompt_mod

log = logging.getLogger("fluxwall.scheduler")


def set_desktop_wallpaper(image_path: str | Path) -> bool:
    """Set the OS desktop wallpaper. Returns True on success, False otherwise."""
    image_path = str(Path(image_path).resolve())
    system = platform.system()
    try:
        if system == "Darwin":
            # POSIX file handles spaces; the path is passed as a discrete -e arg.
            script = f'tell application "System Events" to set picture of every desktop to POSIX file "{image_path}"'
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
            return True
        if system == "Linux":
            uri = f"file://{image_path}"
            subprocess.run(
                ["gsettings", "set", "org.gnome.desktop.background", "picture-uri", uri],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["gsettings", "set", "org.gnome.desktop.background", "picture-uri-dark", uri],
                check=False, capture_output=True,
            )
            return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not set desktop wallpaper: %s", exc)
    return False


def run(config_path: str | None = None, set_desktop: bool | None = None) -> str:
    """Generate today's wallpaper and return its path."""
    from .. import pipeline

    conf = cfg.load_config(config_path)
    sched = conf.scheduler
    prompts = prompt_mod.read_prompt_file(sched.get("theme_rotation", "prompts/wallpapers.txt"))
    prompt = prompt_mod.rotate_daily(prompts)
    log.info("Daily wallpaper prompt: %r", prompt)

    specs = lora_mod.resolve_stack(conf, None)
    specs.extend(cfg.parse_loras(sched.get("loras", [])))

    records = pipeline.generate(
        [prompt],
        profile_name=sched.get("profile", "prod"),
        resolution_name=sched.get("resolution", "uhd"),
        extra_loras=specs,
        config=conf,
    )
    path = records[0].image_path
    want_desktop = sched.get("set_desktop", True) if set_desktop is None else set_desktop
    if want_desktop:
        ok = set_desktop_wallpaper(path)
        log.info("Set desktop wallpaper: %s", ok)
    return path


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(run())
