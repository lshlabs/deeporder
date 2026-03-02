from __future__ import annotations

import time
from collections.abc import Callable

from core_functions.mouse_controller import MouseController
from utils.types import MacroData, PresetData, StepData


class MacroExecutor:
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.mouse = MouseController()

    def _log(self, message: str):
        self.log_callback(message)

    def execute_steps(
        self,
        macro: MacroData,
        preset: PresetData,
        steps: list[StepData],
        stop_requested: Callable[[], bool],
    ) -> None:
        self._log(f"[SUCCESS] 선택된 프리셋: {preset.get('name', preset.get('id', 'preset'))}")

        for step in steps:
            if stop_requested():
                return
            if not step.get("enabled", True):
                continue

            step_type = step.get("step_type")
            if step_type == "delay":
                delay_seconds = float(step.get("delay_sec", 0.0) or 0.0)
                self._log(f"[WAIT] 대기 {delay_seconds:.1f}초")
                time.sleep(delay_seconds)
                continue

            if step_type == "note":
                continue

            item_id = step.get("item_id")
            if not item_id:
                continue
            item = macro.get("items", {}).get(item_id)
            if not item:
                continue

            rect = item.get("screen_rect")
            if not rect:
                self._log(f"[FAIL] {item.get('name', item_id)}: 좌표 정보가 없습니다.")
                continue

            if item.get("item_type") == "text":
                continue

            center_x = int(rect["x"]) + int(rect["width"]) // 2
            center_y = int(rect["y"]) + int(rect["height"]) // 2
            click_count = int(step.get("click_count", 1) or 1)
            self._log(f"[SUCCESS] [버튼] {item.get('name', item_id)} 클릭 x{click_count}")
            self.mouse.click_at(center_x, center_y, clicks=click_count)
