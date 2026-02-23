from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _timestamp_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / "docs" / "portfolio" / "evidence" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def collect_gui_evidence(out_dir: Path) -> dict:
    _ensure_offscreen()
    from PyQt6 import QtCore, QtWidgets
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtTest import QTest

    from dialog.main_dialog import MainDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainDialog()
    window.show()
    app.processEvents()

    # Deterministic timeout path without touching real screen/apps.
    window.macro_runner.image_matcher.reload_templates = lambda: None
    window.macro_runner.image_matcher.find_template = lambda template_id: (False, None, 0.0, None, None)
    window.macro_runner._save_debug_images = lambda *args, **kwargs: None

    selected_macro_key = next(iter(window.macro_name_to_key.values()), None)
    if selected_macro_key:
        window.macro_runner.start_macro(
            selected_macro_key,
            run_options={
                "template_timeout_sec": 1.0,
                "max_retries": 3,
                "retry_interval_sec": 0.2,
                "save_debug_every_n_failures": 999999,
            },
        )

    timeout_deadline = time.time() + 5
    while time.time() < timeout_deadline:
        app.processEvents()
        thread = window.macro_runner.running_macros.get(selected_macro_key) if selected_macro_key else None
        if not thread or not thread.is_alive():
            break
        time.sleep(0.05)

    app.processEvents()
    gui_timeout_png = out_dir / "gui_timeout_log.png"
    window.grab().save(str(gui_timeout_png))

    # F12 emergency-stop capture (window-focused shortcut behavior)
    if selected_macro_key:
        # Use direct log injection so the shortcut effect is visible without starting another worker.
        window.on_log_message("매크로 시작: F12 검증용 더미 실행")
    window.activateWindow()
    app.processEvents()
    QTest.keyClick(window, QtCore.Qt.Key.Key_F12)
    app.processEvents()
    gui_f12_png = out_dir / "gui_f12_log.png"
    window.grab().save(str(gui_f12_png))

    gui_log_txt = out_dir / "gui_log.txt"
    log_text = window.textBrowser_log.toPlainText() if window.textBrowser_log else ""
    gui_log_txt.write_text(log_text, encoding="utf-8")

    result = {
        "macro_used": selected_macro_key,
        "gui_timeout_capture": str(gui_timeout_png.relative_to(ROOT)),
        "gui_f12_capture": str(gui_f12_png.relative_to(ROOT)),
        "gui_log": str(gui_log_txt.relative_to(ROOT)),
    }

    window.close()
    app.processEvents()
    app.quit()
    return result


def _pick_sample(pattern: str) -> Path | None:
    candidates = sorted((ROOT / "test_results").rglob(pattern))
    return candidates[-1] if candidates else None


def collect_delivery_sample_evidence(out_dir: Path) -> dict:
    from image_matcher_easyocr import ImageMatcherEasyOCR

    matcher = ImageMatcherEasyOCR(threshold=0.7)
    samples = {
        "baemin": _pick_sample("baemin_detection.png"),
        "coupang": _pick_sample("coupang_detection.png"),
    }

    report: dict[str, dict] = {}
    for app_name, sample_path in samples.items():
        item: dict[str, object] = {
            "sample_source": str(sample_path.relative_to(ROOT)) if sample_path else None,
            "status": "not_found",
        }
        if not sample_path:
            report[app_name] = item
            continue

        bgr = cv2.imread(str(sample_path))
        if bgr is None:
            item["status"] = "image_load_failed"
            report[app_name] = item
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        copied_src = out_dir / f"{app_name}_sample_source.png"
        cv2.imwrite(str(copied_src), bgr)

        try:
            detected = matcher.detect_delivery_app(rgb, save_image=False)
            roi_buttons = matcher.find_delivery_buttons_by_app(rgb, detected)
            item["status"] = "ok"
            item["detected_app"] = detected
            item["buttons_found_keys"] = sorted(list(roi_buttons.keys())) if isinstance(roi_buttons, dict) else []
        except SystemExit as e:
            item["status"] = "system_exit"
            item["system_exit_code"] = int(getattr(e, "code", 1) or 1)
        except Exception as e:
            item["status"] = "error"
            item["error"] = str(e)

        report[app_name] = item

    (out_dir / "delivery_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main():
    out_dir = _timestamp_dir()
    summary = {"evidence_dir": str(out_dir.relative_to(ROOT))}

    try:
        summary["gui"] = collect_gui_evidence(out_dir)
    except Exception as e:
        summary["gui_error"] = str(e)

    try:
        summary["delivery_samples"] = collect_delivery_sample_evidence(out_dir)
    except Exception as e:
        summary["delivery_samples_error"] = str(e)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
