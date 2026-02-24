from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    # PyInstaller로 실행되면 sys.frozen = True 가 들어간다.
    return bool(getattr(sys, "frozen", False))


def get_base_dir() -> Path:
    # 일반 실행: 프로젝트 루트
    # PyInstaller 실행: 압축 해제된 임시 폴더(_MEIPASS)
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    # ui/data/img 경로 함수를 만들 때 공통으로 쓰는 헬퍼
    return get_base_dir().joinpath(*parts)


def ui_path(filename: str) -> Path:
    return resource_path("ui", filename)


def img_path(*parts: str) -> Path:
    return resource_path("img", *parts)


def data_path() -> Path:
    return resource_path("utils", "data.json")


def debug_dir() -> Path:
    # 디버그 폴더가 없으면 만들어서 저장 코드가 바로 쓸 수 있게 한다.
    path = img_path("debugging")
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_relative_to_base(path: str | Path) -> str:
    p = Path(path)
    base_dir = get_base_dir()
    try:
        return str(p.resolve().relative_to(base_dir.resolve()))
    except Exception:
        # 상대경로 변환이 안 되면 원본 문자열을 그대로 쓴다.
        return str(p)


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value)
    if path.exists():
        return path

    # 프로젝트 상대경로로도 한 번 시도
    candidate = resource_path(str(path).replace("\\", "/"))
    if candidate.exists():
        return candidate

    # 다른 PC에서 저장된 예전 절대경로면 deeporder 뒤 경로만 살려서 다시 붙인다.
    text = str(path).replace("\\", "/")
    for marker in ("/deeporder/", "deeporder/"):
        if marker in text:
            suffix = text.split(marker, 1)[1]
            rebased = resource_path(suffix)
            if rebased.exists():
                return rebased

    # 마지막 fallback: 원본 경로를 그대로 반환해서 호출부에서 로그로 확인 가능하게 한다.
    return path
