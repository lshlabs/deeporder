import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal

from core_functions.mouse_controller import MouseController
from core_functions.trigger_evaluator import TriggerEvaluator
from utils.data_manager import DataManager


class MacroRunner(QObject):
    status_changed = pyqtSignal(str, str)
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running_macros = {}
        self.stop_flags = {}
        self.data_manager = DataManager.get_instance()
        self.default_run_options = {
            "max_retries": 10,
            "retry_interval_sec": 0.5,
            "template_timeout_sec": 10.0,
        }
        try:
            import easyocr

            self.ocr_reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        except Exception:
            self.ocr_reader = None
        self.trigger_evaluator = TriggerEvaluator(self.ocr_reader)

    def _parse_hotkey_combo(self, key_text: str, fallback_text: str):
        alias_map = {
            "CTRL": ("Ctrl", [0x11]),
            "CONTROL": ("Ctrl", [0x11]),
            "ALT": ("Alt", [0x12]),
            "SHIFT": ("Shift", [0x10]),
            "META": ("Meta", [0x5B, 0x5C]),
            "WIN": ("Meta", [0x5B, 0x5C]),
            "WINDOWS": ("Meta", [0x5B, 0x5C]),
        }
        special_keys = {
            "TAB": ("Tab", 0x09),
            "SPACE": ("Space", 0x20),
            "ENTER": ("Enter", 0x0D),
            "RETURN": ("Enter", 0x0D),
            "ESC": ("Esc", 0x1B),
            "ESCAPE": ("Esc", 0x1B),
        }

        tokens = [token.strip() for token in str(key_text or "").split("+") if token.strip()]
        if not tokens:
            tokens = [fallback_text]

        display_parts = []
        modifiers = []
        primary = None

        for token in tokens:
            normalized = token.upper()
            if normalized in alias_map:
                label, group = alias_map[normalized]
                if label not in display_parts:
                    display_parts.append(label)
                    modifiers.append(group)
                continue

            if normalized.startswith("F") and normalized[1:].isdigit():
                number = int(normalized[1:])
                if 1 <= number <= 24:
                    display_parts.append(f"F{number}")
                    primary = [0x6F + number]
                    continue

            if normalized in special_keys:
                label, vk = special_keys[normalized]
                display_parts.append(label)
                primary = [vk]
                continue

            if len(normalized) == 1:
                code = ord(normalized)
                if 48 <= code <= 57 or 65 <= code <= 90:
                    display_parts.append(normalized)
                    primary = [code]

        if primary is None:
            if str(key_text or "").strip().upper() == str(fallback_text or "").strip().upper():
                return {"text": "F12", "modifiers": [], "primary": [0x7B]}
            return self._parse_hotkey_combo(fallback_text, fallback_text)

        return {"text": "+".join(display_parts), "modifiers": modifiers, "primary": primary}

    def _stop_hotkey_pressed(self):
        settings = self.data_manager._data.get("settings_main", {})
        combo = self._parse_hotkey_combo(settings.get("stop_hotkey"), "F12")
        try:
            import ctypes

            user32 = ctypes.windll.user32
            for group in combo.get("modifiers", []):
                if not any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in group):
                    return False
            return any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in combo.get("primary", []))
        except Exception:
            return False

    def _emit_log(self, message):
        self.log_message.emit(str(message))

    def _format_rect_point(self, rect: dict | None):
        if not rect:
            return "(0, 0)"
        x = int(rect.get("x", 0)) + int(rect.get("width", 1)) // 2
        y = int(rect.get("y", 0)) + int(rect.get("height", 1)) // 2
        return f"({x}, {y})"

    def _log_macro_trigger_wait(self, macro_name: str, macro_trigger: dict | None):
        if not macro_trigger or not macro_trigger.get("enabled"):
            return
        if macro_trigger.get("trigger_type") == "pixel_color":
            color = macro_trigger.get("expected_color") or {}
            self._emit_log(f"[WAIT] [{macro_name}] 트리거 대기 중(설정 컬러: {color})")
        elif macro_trigger.get("trigger_type") == "text_roi":
            text = str(macro_trigger.get("expected_text") or "").strip()
            self._emit_log(f"[WAIT] [{macro_name}] 트리거 대기 중(설정 텍스트: '{text}')")

    def _log_macro_trigger_found(self, macro_name: str, macro_trigger: dict | None):
        if not macro_trigger or not macro_trigger.get("enabled"):
            return
        rect = macro_trigger.get("screen_rect")
        point = self._format_rect_point(rect)
        if macro_trigger.get("trigger_type") == "pixel_color":
            color = macro_trigger.get("expected_color") or {}
            self._emit_log(f"[SUCCESS] [{macro_name}] 트리거 발견(설정 컬러: {color}) at {point}")
        elif macro_trigger.get("trigger_type") == "text_roi":
            text = str(macro_trigger.get("expected_text") or "").strip()
            self._emit_log(f"[SUCCESS] [{macro_name}] 트리거 발견(설정 텍스트: '{text}') at {point}")

    def _log_preset_trigger_found(self, preset_name: str, preset_trigger: dict | None):
        if not preset_trigger or not preset_trigger.get("enabled"):
            return
        rect = preset_trigger.get("screen_rect")
        point = self._format_rect_point(rect)
        if preset_trigger.get("trigger_type") == "text_roi":
            text = str(preset_trigger.get("expected_text") or "").strip()
            self._emit_log(f"[SUCCESS] [{preset_name}] 트리거 발견(설정 텍스트: '{text}') at {point}")
        elif preset_trigger.get("trigger_type") == "pixel_color":
            color = preset_trigger.get("expected_color") or {}
            self._emit_log(f"[SUCCESS] [{preset_name}] 트리거 발견(설정 컬러: {color}) at {point}")

    def start_macro(self, macro_key, run_options=None):
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
            thread = threading.Thread(target=self._run_macro, args=(macro_key, stop_event))
            thread.daemon = True
            thread.start()
            self.running_macros[macro_key] = thread
            self.status_changed.emit(macro_key, "running")
            return True
        except Exception as e:
            self._emit_log(f"[FAIL] 매크로 시작 실패: {e}")
            self.stop_flags.pop(macro_key, None)
            self.running_macros.pop(macro_key, None)
            return False

    def stop_macro(self, macro_key):
        stop_event = self.stop_flags.get(macro_key)
        if stop_event is None:
            return False
        stop_event.set()
        thread = self.running_macros.get(macro_key)
        if thread is not None:
            thread.join(1.0)
        self.stop_flags.pop(macro_key, None)
        self.running_macros.pop(macro_key, None)
        self.status_changed.emit(macro_key, "stopped")
        return True

    def stop_all(self):
        for macro_key in list(self.stop_flags.keys()):
            self.stop_macro(macro_key)

    def _execute_preset_steps(self, macro_key, macro, preset_id, frame, stop_flag):
        preset = macro.get("presets", {}).get(preset_id)
        if not preset:
            self._emit_log("[FAIL] 실행 가능한 프리셋을 찾지 못했습니다.")
            return

        self._emit_log(f"[SUCCESS] 선택된 프리셋: {preset.get('name', preset_id)}")
        mouse = MouseController()
        steps = self.data_manager.sort_preset_steps(macro_key, preset_id)

        for step in steps:
            if stop_flag.is_set():
                break
            if self._stop_hotkey_pressed():
                stop_flag.set()
                break
            if not step.get("enabled", True):
                continue

            if step.get("step_type") == "delay":
                delay_sec = float(step.get("delay_sec", 0.0) or 0.0)
                self._emit_log(f"[WAIT] 대기 {delay_sec:.1f}초")
                time.sleep(delay_sec)
                continue

            if step.get("step_type") == "note":
                continue

            item = macro.get("items", {}).get(step.get("item_id"))
            if not item:
                continue

            rect = item.get("screen_rect")
            if not rect:
                self._emit_log(f"[FAIL] {item.get('name', step.get('item_id'))}: 좌표가 없습니다.")
                continue

            if item.get("item_type") == "text":
                continue

            center_x = int(rect["x"]) + int(rect["width"]) // 2
            center_y = int(rect["y"]) + int(rect["height"]) // 2
            click_count = int(step.get("click_count", 1) or 1)
            self._emit_log(f"[SUCCESS] [버튼] {item.get('name')} 클릭 x{click_count}")
            mouse.click_at(center_x, center_y, clicks=click_count)

    def _run_macro(self, macro_key, stop_flag):
        macro = self.data_manager.get_macro(macro_key)
        if not macro:
            return

        settings = self.data_manager._data.get("settings_main", {})
        stop_hotkey = str(settings.get("stop_hotkey") or "F12")
        macro_settings = macro.get("settings", {}) or {}
        repeat_count = int(macro_settings.get("repeat_count", 1) or 0)
        repeat_delay = float(macro_settings.get("repeat_delay_sec", 0.5) or 0.0)
        completed_cycles = 0
        macro_name = macro.get("name", macro_key)
        macro_trigger = macro.get("macro_trigger")
        default_preset_id = macro.get("default_preset_id")

        self._emit_log(f"[START] 감시모드 시작: {macro_name} (종료 핫키: {stop_hotkey})")
        last_selected_preset = None
        last_trigger_blocked = False

        try:
            while not stop_flag.is_set():
                if self._stop_hotkey_pressed():
                    self._emit_log(f"[END] {stop_hotkey} 감지: 감시모드를 종료합니다.")
                    stop_flag.set()
                    break

                preset_id, frame, macro_meta = self.trigger_evaluator.choose_preset(macro, default_preset_id)
                if preset_id is None:
                    if not last_trigger_blocked:
                        self._log_macro_trigger_wait(macro_name, macro_trigger)
                    last_trigger_blocked = True
                    last_selected_preset = None
                    time.sleep(0.2)
                    continue

                if last_trigger_blocked:
                    self._log_macro_trigger_found(macro_name, macro_trigger)

                last_trigger_blocked = False
                if preset_id != last_selected_preset:
                    preset = macro.get("presets", {}).get(preset_id, {})
                    preset_name = preset.get("name", preset_id)
                    next_cycle = completed_cycles + 1

                    if preset_id != default_preset_id:
                        self._log_preset_trigger_found(preset_name, preset.get("preset_trigger"))

                    if repeat_count == 0:
                        self._emit_log(f"[START] 반복 실행 {next_cycle}회차")
                    elif repeat_count > 1:
                        self._emit_log(f"[START] 반복 실행 {next_cycle}/{repeat_count}회차")

                    self._execute_preset_steps(macro_key, macro, preset_id, frame, stop_flag)
                    if stop_flag.is_set():
                        break

                    completed_cycles += 1
                    last_selected_preset = preset_id

                    if repeat_count != 0 and completed_cycles >= max(1, repeat_count):
                        self._emit_log(f"[END] 설정된 반복 실행 {repeat_count}회가 완료되어 감시모드를 종료합니다.")
                        stop_flag.set()
                        break

                    if repeat_delay > 0:
                        self._emit_log(f"[WAIT] 다음 반복 대기중({repeat_delay:.1f}초)")
                        time.sleep(repeat_delay)

                time.sleep(0.2)

        except Exception as e:
            self._emit_log(f"[FAIL] 매크로 실행 중 오류: {e}")
        finally:
            self._emit_log(f"[END] 감시모드 종료: {macro_name}")
            self.stop_flags.pop(macro_key, None)
            self.running_macros.pop(macro_key, None)
            self.status_changed.emit(macro_key, "stopped")
