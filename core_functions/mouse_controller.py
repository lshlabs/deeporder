from __future__ import annotations

import time
from collections.abc import Mapping

import pyautogui


class MouseController:
    def __init__(self, click_delay: float = 0.1) -> None:
        self.click_delay = float(click_delay)
        pyautogui.PAUSE = self.click_delay

    def click_at(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
        delay: float | None = None,
    ) -> bool:
        wait_time = self.click_delay if delay is None else max(0.0, float(delay))
        pyautogui.moveTo(x, y)
        time.sleep(wait_time)
        pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=wait_time)
        time.sleep(wait_time)
        return True

    def click_at_template(
        self,
        matcher,
        template_id: str,
        button: str = "left",
        clicks: int = 1,
        offset: tuple[int, int] = (0, 0),
    ) -> bool:
        success, location, _, _, _ = matcher.find_template(template_id)
        if not success:
            return False

        template = matcher.load_template(template_id)
        if template is None:
            return False

        height, width = template.shape[:2]
        return self.click_at(
            location[0] + width // 2 + offset[0],
            location[1] + height // 2 + offset[1],
            button=button,
            clicks=clicks,
        )

    def click_at_action(
        self,
        matcher,
        template_id: str,
        action_id: str,
        button: str = "left",
        clicks: int = 1,
        offset: tuple[int, int] = (0, 0),
    ) -> bool:
        success, location, _, _, scale_info = matcher.find_template(template_id)
        if not success:
            return False

        center = matcher.get_action_center(template_id, action_id, location, scale_info)
        if center is None:
            return False

        return self.click_at(center[0] + offset[0], center[1] + offset[1], button=button, clicks=clicks)

    def click_all_actions(
        self,
        matcher,
        template_id: str,
        fixed_location=None,
        fixed_scale_info=None,
    ) -> tuple[int, int]:
        if fixed_location is None or fixed_scale_info is None:
            success, location, _, _, scale_info = matcher.find_template(template_id)
            if not success:
                return 0, 0
        else:
            location = fixed_location
            scale_info = fixed_scale_info

        template_actions: Mapping[str, dict] = matcher.template_actions.get(template_id, {})
        if not template_actions:
            return 0, 0

        success_count = 0
        failure_count = 0
        sorted_actions = sorted(template_actions.items(), key=lambda item: item[1].get("number", 0))

        for action_id, action_data in sorted_actions:
            if not action_data.get("enabled", True):
                continue
            center = matcher.get_action_center(template_id, action_id, location, scale_info)
            if center is None:
                failure_count += 1
                continue
            if self.click_at(center[0], center[1]):
                success_count += 1
            else:
                failure_count += 1

        return success_count, failure_count
