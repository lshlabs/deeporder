@echo off
setlocal

REM Run from repository root on Windows
if not exist .venv (
  py -3 -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements-dev.txt

REM EasyOCR model can be pre-downloaded before packaging for offline use.
pyinstaller --noconfirm --clean DeepOrder.spec

echo Build complete. Output: dist\DeepOrder\ or dist\DeepOrder.exe (depending on PyInstaller version/settings)
endlocal
