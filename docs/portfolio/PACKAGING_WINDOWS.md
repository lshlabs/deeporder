# Windows Packaging Guide (Preparation-Only in Current macOS Environment)

## Goal

Prepare a reproducible Windows `.exe` build path for DeepOrder using PyInstaller.

## Included Artifacts

- `DeepOrder.spec`
- `scripts/build_windows.bat`
- `main.py` (packaging entrypoint)

## Why `main.py`

Using a root-level entrypoint keeps PyInstaller configuration stable and avoids path issues from invoking `dialog/main_dialog.py` directly.

## Prerequisites (Windows Host)

- Windows 10/11
- Python 3.x
- Visual C++ Redistributable (if required by native deps)
- Internet access for initial `easyocr` model download (or pre-download/copy model cache)

## Build Steps (Windows)

1. Open `cmd.exe` in the repository root.
2. Run `scripts\build_windows.bat`.
3. Check the `dist\` output directory.

## Asset / Data Inclusion Strategy

The spec includes:

- `ui/` (Qt `.ui` assets)
- `img/` (runtime image assets; excludes `img/debugging` outputs)
- `utils/data.json`

## Hidden Imports

The spec explicitly includes:

- `easyocr`
- `cv2`
- `numpy`
- `mss`
- `pyautogui`

It also collects EasyOCR submodules to reduce missing-import runtime failures.

## Path Handling (PyInstaller)

Runtime resource access is routed through `utils/path_manager.py` and supports:

- Script mode (`Path(__file__).resolve().parent.parent`)
- PyInstaller mode (`sys._MEIPASS`)

This is the key change that makes packaging practical.

## Notes / Limitations

- This repository was prepared on macOS; Windows `.exe` was not built in this pass.
- EasyOCR model files may still require validation for fully offline distribution.
- If packaging size is too large, split “runtime assets” and “debug/test assets” more aggressively.
