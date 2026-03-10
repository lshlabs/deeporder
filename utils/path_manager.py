from __future__ import annotations

import shutil
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_resource_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent


def get_runtime_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    return get_resource_dir().joinpath(*parts)


def runtime_path(*parts: str) -> Path:
    return get_runtime_dir().joinpath(*parts)


def ui_path(filename: str) -> Path:
    return resource_path("ui", filename)


def _copy_seed_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or not source.exists():
        return
    shutil.copy2(source, target)


def _copy_seed_tree(source_root: Path, target_root: Path) -> None:
    if target_root.exists() or not source_root.exists():
        return
    shutil.copytree(source_root, target_root)


def img_path(*parts: str) -> Path:
    runtime_root = runtime_path("img")
    bundled_root = resource_path("img")
    _copy_seed_tree(bundled_root, runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root.joinpath(*parts)


def data_path() -> Path:
    runtime_file = runtime_path("utils", "data.json")
    bundled_file = resource_path("utils", "data.json")
    _copy_seed_file(bundled_file, runtime_file)
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    return runtime_file


def debug_dir() -> Path:
    path = img_path("debugging")
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_relative_to_base(path: str | Path) -> str:
    current_path = Path(path)
    resource_base = get_resource_dir()
    runtime_base = get_runtime_dir()

    for base_dir in (runtime_base, resource_base):
        try:
            return str(current_path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            continue

    return str(current_path)


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value)
    if path.exists():
        return path

    normalized = str(path).replace("\\", "/")
    runtime_candidate = runtime_path(normalized)
    if runtime_candidate.exists():
        return runtime_candidate

    resource_candidate = resource_path(normalized)
    if resource_candidate.exists():
        return resource_candidate

    for marker in ("/deeporder/", "deeporder/"):
        if marker not in normalized:
            continue
        suffix = normalized.split(marker, 1)[1]

        rebased_runtime = runtime_path(suffix)
        if rebased_runtime.exists():
            return rebased_runtime

        rebased_resource = resource_path(suffix)
        if rebased_resource.exists():
            return rebased_resource

    return path
