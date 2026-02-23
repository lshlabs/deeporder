# Validation Scenarios (Manual / Portfolio Evidence)

## Latest Evidence Snapshot (2026-02-23)

- Evidence directory: `docs/portfolio/evidence/20260223_220503`
- GUI captures:
  - `docs/portfolio/evidence/20260223_220503/gui_timeout_log.png`
  - `docs/portfolio/evidence/20260223_220503/gui_f12_log.png`
  - `docs/portfolio/evidence/20260223_220503/gui_log.txt`
- Delivery sample validation:
  - 배민 샘플 성공: `docs/portfolio/evidence/20260223_220503/delivery_validation_report.json`
  - 쿠팡 샘플 추가 성공: `docs/portfolio/evidence/20260223_220503/delivery_validation_coupang_retry.json`

참고:
- 이번 증빙은 현재 세션 환경 제약상 **실배달앱 라이브 실행**이 아니라 저장소 내 샘플 스크린샷 기반 OCR 검증 + `main.py` UI 오프스크린 캡처로 수집했습니다.

## 1. Startup / Path Stability

1. Run `python main.py` from the repository root.
2. Run `python /absolute/path/to/deeporder/main.py` from a different current directory.
3. Confirm main window opens and no `ui/...` path error occurs.

Expected:
- Main window loads in both cases.
- `utils/data.json` is read successfully.

## 2. GUI Log Panel

1. Start the app.
2. Trigger a macro run or stop action.
3. Observe `textBrowser_log` in the main window.

Expected:
- Runtime messages appear in GUI log panel and remain capped (rolling window behavior).
- Console output may still mirror logs for debugging.

## 3. Timeout / Retry Guard

1. Use a macro whose original template image is not currently visible on screen.
2. Start the macro.

Expected:
- Log shows retry progression.
- Execution stops after timeout or max retries.
- UI status returns from `(실행 중)` to normal.

## 4. Emergency Stop (F12)

1. Start one or more macros.
2. Focus the main app window.
3. Press `F12`.

Expected:
- Running macro stop flags are set.
- GUI log shows emergency stop request.
- Status labels/items recover to stopped state.

## 5. Vision Engine Abstraction (Code-Level Check)

1. Inspect `core_functions/vision_engine.py`.
2. Confirm `MacroRunner` depends on `VisionEngine`, not directly on `ImageMatcherEasyOCR`.

Expected:
- `VisionEngine(mode="ocr"|"template")` exists.
- Template metadata / action coordinates remain available through the facade.

## 6. Packaging Readiness (Windows Preparation)

1. Review `DeepOrder.spec`.
2. Review `scripts/build_windows.bat`.
3. Review `docs/portfolio/PACKAGING_WINDOWS.md`.

Expected:
- UI/assets/data are listed in PyInstaller datas.
- Hidden imports for `easyocr`, `cv2`, `numpy`, `mss`, `pyautogui` are defined.
- Windows build steps are documented and reproducible on a Windows host.
