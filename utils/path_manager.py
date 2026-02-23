from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_base_dir() -> Path:
    """
    Return the project root in script mode and the unpacked resource root in
    PyInstaller mode.
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    return get_base_dir().joinpath(*parts)


def ui_path(filename: str) -> Path:
    return resource_path("ui", filename)


def img_path(*parts: str) -> Path:
    return resource_path("img", *parts)


def data_path() -> Path:
    return resource_path("utils", "data.json")


def debug_dir() -> Path:
    path = img_path("debugging")
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_relative_to_base(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(get_base_dir().resolve()))
    except Exception:
        return str(p)


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    candidate = resource_path(str(path).replace("\\", "/"))
    if candidate.exists():
        return candidate
    # Handle legacy absolute paths from another machine by preserving suffix from project root markers.
    text = str(path).replace("\\", "/")
    for marker in ("/deeporder/", "deeporder/"):
        if marker in text:
            suffix = text.split(marker, 1)[1]
            rebased = resource_path(suffix)
            if rebased.exists():
                return rebased
    return path
