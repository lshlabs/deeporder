from __future__ import annotations

import mss
import numpy as np


class TriggerEvaluator:
    def __init__(self, ocr_reader=None):
        self.ocr_reader = ocr_reader

    def capture_screen(self):
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            frame = np.array(sct.grab(monitor))
        return frame[:, :, :3]

    def evaluate(self, trigger: dict | None, frame=None):
        if not trigger or not trigger.get("enabled"):
            return True, None

        frame = frame if frame is not None else self.capture_screen()
        trigger_type = trigger.get("trigger_type")
        if trigger_type == "pixel_color":
            return self._eval_pixel_color(trigger, frame)
        if trigger_type == "text_roi":
            return self._eval_text_roi(trigger, frame)
        return False, None

    def _crop(self, rect: dict, frame):
        x = max(0, int(rect.get("x", 0)))
        y = max(0, int(rect.get("y", 0)))
        width = max(1, int(rect.get("width", 1)))
        height = max(1, int(rect.get("height", 1)))
        return frame[y : y + height, x : x + width]

    def _eval_pixel_color(self, trigger: dict, frame):
        rect = trigger.get("screen_rect")
        color = trigger.get("expected_color")
        if not rect or not color:
            return False, None

        x = max(0, int(rect.get("x", 0)) + int(rect.get("width", 1)) // 2)
        y = max(0, int(rect.get("y", 0)) + int(rect.get("height", 1)) // 2)
        y = min(y, frame.shape[0] - 1)
        x = min(x, frame.shape[1] - 1)
        pixel = frame[y, x]
        tolerance = int(trigger.get("color_tolerance", 10) or 10)
        expected = [int(color.get(channel, 0)) for channel in ("r", "g", "b")]
        actual = [int(pixel[2]), int(pixel[1]), int(pixel[0])]
        matched = all(abs(actual[i] - expected[i]) <= tolerance for i in range(3))
        return matched, {"actual_color": {"r": actual[0], "g": actual[1], "b": actual[2]}}

    def _eval_text_roi(self, trigger: dict, frame):
        rect = trigger.get("screen_rect")
        expected_text = str(trigger.get("expected_text") or "").strip()
        if not rect or not expected_text or self.ocr_reader is None:
            return False, {"detected_text": ""}

        roi = self._crop(rect, frame)
        results = self.ocr_reader.readtext(roi, paragraph=False)
        detected_text = " ".join(str(result[1]).strip() for result in results if len(result) >= 3).strip()
        min_conf = float(trigger.get("ocr_confidence_min", 0.6) or 0.6)
        confident_texts = [str(result[1]).strip() for result in results if len(result) >= 3 and float(result[2]) >= min_conf]
        confident_joined = " ".join(confident_texts).strip()
        haystack = confident_joined if confident_joined else detected_text

        if trigger.get("text_match_mode") == "exact":
            matched = haystack == expected_text
        else:
            matched = expected_text in haystack
        return matched, {"detected_text": haystack}

    def choose_preset(self, macro: dict, default_preset_id: str):
        frame = self.capture_screen()
        macro_ok, macro_meta = self.evaluate(macro.get("macro_trigger"), frame)
        if not macro_ok:
            return None, frame, macro_meta

        for preset_id, preset in macro.get("presets", {}).items():
            if preset_id == default_preset_id:
                continue
            matched, _ = self.evaluate(preset.get("preset_trigger"), frame)
            if matched:
                return preset_id, frame, macro_meta
        return default_preset_id, frame, macro_meta
