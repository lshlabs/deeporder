import os
import sys
from pathlib import Path


def _prepend_dll_dir(path: Path) -> None:
    if not path.exists():
        return
    try:
        os.add_dll_directory(str(path))
    except Exception:
        pass
    os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    base = Path(getattr(sys, "_MEIPASS", ""))
    if str(base):
        _prepend_dll_dir(base / "torch" / "lib")
        _prepend_dll_dir(base / "PyQt6" / "Qt6" / "bin")

    try:
        import torch  # noqa: F401
    except Exception as exc:
        print(f"[runtime_hook] torch preload failed: {exc}", file=sys.stderr)
