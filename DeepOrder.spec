# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path(globals().get("SPECPATH", ".")).resolve()


def data_entries():
    datas = []

    config_path = project_root / "data" / "config.v2.json"
    if config_path.exists():
        datas.append((str(config_path), "data"))

    ui_root = project_root / "ui"
    if ui_root.exists():
        datas.append((str(ui_root), "ui"))

    user_data_path = project_root / "utils" / "data.json"
    if user_data_path.exists():
        datas.append((str(user_data_path), "utils"))

    img_root = project_root / "img"
    if img_root.exists():
        for path in img_root.rglob("*"):
            if not path.is_file():
                continue
            if "debugging" in path.parts:
                continue
            datas.append((str(path), str(Path("img") / path.relative_to(img_root).parent)))

    resources_root = project_root / "resources"
    if resources_root.exists():
        datas.append((str(resources_root), "resources"))

    return datas


hiddenimports = [
    "easyocr",
    "cv2",
    "numpy",
    "mss",
    "pyautogui",
] + collect_submodules("easyocr")

datas = data_entries() + collect_data_files("easyocr")


a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_root / "runtime_hooks" / "preload_torch.py")],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DeepOrder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DeepOrder",
)
