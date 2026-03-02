from __future__ import annotations

import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal

from core_functions.hotkey_monitor import HotkeyMonitor
from core_functions.macro_executor import MacroExecutor
from core_functions.trigger_evaluator import TriggerEvaluator
from utils.data_manager import DataManager
from utils.types import MacroData, PresetData, TriggerData


class MacroRunner(QObject):
    status_changed = pyqtSignal(str, str)
    log_message = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.running_macros: dict[str, threading.Thread] = {}
        self.stop_flags: dict[str, threading.Event] = {}
        self.data_manager = DataManager.get_instance()
        self.default_run_options = {
            "max_retries": 10,
            "retry_interval_sec": 0.5,
            "template_timeout_sec": 10.0,
        }
        self.ocr_reader = self._build_ocr_reader()
        self.trigger_evaluator = TriggerEvaluator(self.ocr_reader)
        self.macro_executor = MacroExecutor(self._emit_log)

    def _build_ocr_reader(self):
        try:
            import easyocr

            return easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        except Exception:
            return None

    def _emit_log(self, message: str) -> None:
        self.log_message.emit(str(message))

    def _format_rect_point(self, rect: dict | None) -> str:
        if not rect:
            return "(0, 0)"
        center_x = int(rect.get("x", 0)) + int(rect.get("width", 1)) // 2
        center_y = int(rect.get("y", 0)) + int(rect.get("height", 1)) // 2
        return f"({center_x}, {center_y})"

    def _format_trigger_summary(self, trigger: TriggerData | dict | None) -> str:
        if not trigger:
            return ""
        trigger_type = trigger.get("trigger_type")
        if trigger_type == "pixel_color":
            return f"설정 컬러: {trigger.get('expected_color') or {}}"
        if trigger_type == "text_roi":
            expected_text = str(trigger.get("expected_text") or "").strip()
            return f"설정 텍스트: '{expected_text}'"
        return "설정값 없음"

    def _log_trigger_wait(self, macro_name: str, macro_trigger: TriggerData | dict | None) -> None:
        if not macro_trigger or not macro_trigger.get("enabled"):
            return
        summary = self._format_trigger_summary(macro_trigger)
        self._emit_log(f"[WAIT] [{macro_name}] 트리거 대기 중({summary})")

    def _log_trigger_found(self, owner_name: str, trigger: TriggerData | dict | None) -> None:
        if not trigger or not trigger.get("enabled"):
            return
        summary = self._format_trigger_summary(trigger)
        point_text = self._format_rect_point(trigger.get("screen_rect"))
        self._emit_log(f"[SUCCESS] [{owner_name}] 트리거 발견({summary}) at {point_text}")

    def _log_monitor_start(self, macro_name: str, stop_hotkey: str) -> None:
        self._emit_log(f"[START] 감시모드 시작: {macro_name} (종료 핫키: {stop_hotkey})")

    def _log_monitor_end(self, macro_name: str) -> None:
        self._emit_log(f"[END] 감시모드 종료: {macro_name}")

    def _log_repeat_start(self, completed_cycles: int, repeat_count: int) -> None:
        next_cycle = completed_cycles + 1
        if repeat_count == 0:
            self._emit_log(f"[START] 반복 실행 {next_cycle}회차")
            return
        if repeat_count > 1:
            self._emit_log(f"[START] 반복 실행 {next_cycle}/{repeat_count}회차")

    def _log_repeat_wait(self, repeat_delay: float) -> None:
        if repeat_delay > 0:
            self._emit_log(f"[WAIT] 다음 반복 대기중({repeat_delay:.1f}초)")

    def _should_stop_by_hotkey(self, stop_monitor: HotkeyMonitor) -> bool:
        return stop_monitor.is_pressed()

    def _handle_stop_hotkey(self, stop_flag: threading.Event, stop_monitor: HotkeyMonitor) -> None:
        self._emit_log(f"[END] {stop_monitor.text} 감지: 감시모드를 종료합니다.")
        stop_flag.set()

    def _is_repeat_limit_reached(self, completed_cycles: int, repeat_count: int) -> bool:
        if repeat_count == 0:
            return False
        return completed_cycles >= max(1, repeat_count)

    def _log_repeat_limit_end(self, repeat_count: int) -> None:
        self._emit_log(f"[END] 설정된 반복 실행 {repeat_count}회가 완료되어 감시모드를 종료합니다.")

    def _load_stop_monitor(self) -> HotkeyMonitor:
        return HotkeyMonitor(self.data_manager.get_stop_hotkey(), "F12")

    def start_macro(self, macro_key: str, run_options: dict | None = None) -> bool:
        try:
            if macro_key in self.running_macros and self.running_macros[macro_key].is_alive():
                self._emit_log(f"[FAIL] 이미 실행 중인 매크로입니다: {macro_key}")
                return False

            macro_data = self.data_manager.get_macro(macro_key)
            if not macro_data:
                self._emit_log(f"[FAIL] 매크로를 찾을 수 없습니다: {macro_key}")
                return False

            stop_event = threading.Event()
            self.stop_flags[macro_key] = stop_event
            worker = threading.Thread(target=self._run_macro, args=(macro_key, stop_event), daemon=True)
            worker.start()
            self.running_macros[macro_key] = worker
            self.status_changed.emit(macro_key, "running")
            return True
        except Exception as exc:
            self._emit_log(f"[FAIL] 매크로 시작 실패: {exc}")
            self.stop_flags.pop(macro_key, None)
            self.running_macros.pop(macro_key, None)
            return False

    def stop_macro(self, macro_key: str) -> bool:
        stop_event = self.stop_flags.get(macro_key)
        if stop_event is None:
            return False
        stop_event.set()
        worker = self.running_macros.get(macro_key)
        if worker is not None:
            worker.join(1.0)
        self.stop_flags.pop(macro_key, None)
        self.running_macros.pop(macro_key, None)
        self.status_changed.emit(macro_key, "stopped")
        return True

    def stop_all(self) -> None:
        for macro_key in list(self.stop_flags.keys()):
            self.stop_macro(macro_key)

    def _execute_selected_preset(
        self,
        macro_key: str,
        macro: MacroData,
        preset_id: str,
        stop_flag: threading.Event,
        stop_monitor: HotkeyMonitor,
    ) -> None:
        preset = macro.get("presets", {}).get(preset_id)
        if not preset:
            self._emit_log("[FAIL] 실행 가능한 프리셋을 찾지 못했습니다.")
            return
        steps = self.data_manager.sort_preset_steps(macro_key, preset_id)
        self.macro_executor.execute_steps(
            macro,
            preset,
            steps,
            lambda: stop_flag.is_set() or self._should_stop_by_hotkey(stop_monitor),
        )
        if self._should_stop_by_hotkey(stop_monitor):
            self._handle_stop_hotkey(stop_flag, stop_monitor)

    def _run_one_monitor_cycle(
        self,
        macro_key: str,
        macro: MacroData,
        stop_flag: threading.Event,
        stop_monitor: HotkeyMonitor,
        repeat_count: int,
        repeat_delay: float,
        completed_cycles: int,
        last_selected_preset: str | None,
        last_trigger_blocked: bool,
    ) -> tuple[int, str | None, bool]:
        default_preset_id = str(macro.get("default_preset_id") or "")
        macro_name = str(macro.get("name") or macro_key)
        macro_trigger = macro.get("macro_trigger")

        preset_id, _frame, _meta = self.trigger_evaluator.choose_preset(macro, default_preset_id)
        if preset_id is None:
            if not last_trigger_blocked:
                self._log_trigger_wait(macro_name, macro_trigger)
            return completed_cycles, None, True

        if last_trigger_blocked:
            self._log_trigger_found(macro_name, macro_trigger)

        if preset_id == last_selected_preset:
            return completed_cycles, last_selected_preset, False

        preset: PresetData = macro.get("presets", {}).get(preset_id, {})
        preset_name = str(preset.get("name") or preset_id)
        if preset_id != default_preset_id:
            self._log_trigger_found(preset_name, preset.get("preset_trigger"))

        self._log_repeat_start(completed_cycles, repeat_count)
        self._execute_selected_preset(macro_key, macro, preset_id, stop_flag, stop_monitor)
        if stop_flag.is_set():
            return completed_cycles, preset_id, False

        updated_cycles = completed_cycles + 1
        if self._is_repeat_limit_reached(updated_cycles, repeat_count):
            self._log_repeat_limit_end(repeat_count)
            stop_flag.set()
            return updated_cycles, preset_id, False

        if repeat_delay > 0:
            self._log_repeat_wait(repeat_delay)
            time.sleep(repeat_delay)

        return updated_cycles, preset_id, False

    def _run_macro(self, macro_key: str, stop_flag: threading.Event) -> None:
        macro = self.data_manager.get_macro(macro_key)
        if not macro:
            return

        macro_name = str(macro.get("name") or macro_key)
        macro_settings = macro.get("settings", {}) or {}
        repeat_count = int(macro_settings.get("repeat_count", 1) or 0)
        repeat_delay = float(macro_settings.get("repeat_delay_sec", 0.5) or 0.0)
        completed_cycles = 0
        last_selected_preset: str | None = None
        last_trigger_blocked = False
        stop_monitor = self._load_stop_monitor()

        self._log_monitor_start(macro_name, stop_monitor.text)
        try:
            while not stop_flag.is_set():
                if self._should_stop_by_hotkey(stop_monitor):
                    self._handle_stop_hotkey(stop_flag, stop_monitor)
                    break

                completed_cycles, last_selected_preset, last_trigger_blocked = self._run_one_monitor_cycle(
                    macro_key=macro_key,
                    macro=macro,
                    stop_flag=stop_flag,
                    stop_monitor=stop_monitor,
                    repeat_count=repeat_count,
                    repeat_delay=repeat_delay,
                    completed_cycles=completed_cycles,
                    last_selected_preset=last_selected_preset,
                    last_trigger_blocked=last_trigger_blocked,
                )

                if stop_flag.is_set():
                    break

                time.sleep(0.2)
        except Exception as exc:
            self._emit_log(f"[FAIL] 매크로 실행 중 오류: {exc}")
        finally:
            self._log_monitor_end(macro_name)
            self.stop_flags.pop(macro_key, None)
            self.running_macros.pop(macro_key, None)
            self.status_changed.emit(macro_key, "stopped")
