# DeepOrder Before / After (Portfolio Summary)

## As-Is (Before)

- `uic.loadUi('ui/...')` and debug/image paths were hardcoded and depended on the current working directory.
- `MacroRunner` was directly coupled to a concrete OCR matcher implementation.
- Execution logs were printed to console only (`dialog/main_dialog.py:on_log_message`).
- Retry behavior was fixed and opaque (no configurable timeout/retry policy per macro).
- Experimental scripts and manual test scripts were mixed with runtime entry files in the project root.

## To-Be (Implemented)

- Added `utils/path_manager.py` and routed UI/data/debug paths through a single path API (`ui_path`, `data_path`, `debug_dir`, etc.).
- Added `core_functions/vision_engine.py` as a lightweight strategy facade over template/OCR matchers.
- Updated `MacroRunner` to use `VisionEngine`, optional `run_options`, timeout/retry settings, and safe cleanup in `finally`.
- Added a GUI log panel (`textBrowser_log`) to `ui/MainWindow.ui` and connected runtime logs to the main window via `utils/logger_ui.py`.
- Added `F12` (main-window focus) emergency stop shortcut.
- Split scripts into `experiments/` and `tests/manual/` for clearer repository structure.
- Added Windows packaging preparation artifacts: `DeepOrder.spec` and `scripts/build_windows.bat`.

## Result (What This Demonstrates)

- Environment-independent resource loading (script mode / PyInstaller-ready path resolution).
- Better observability: users can see retries, timeouts, and failure reasons in the GUI.
- Cleaner architecture for future matcher expansion (OCR/template strategy boundary).
- Safer runtime behavior with timeout/retry controls and emergency stop.
- Stronger portfolio narrative: problem definition, targeted refactor, and deployment preparation.
