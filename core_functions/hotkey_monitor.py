from __future__ import annotations

from utils.hotkey_utils import is_hotkey_pressed, parse_hotkey_text


class HotkeyMonitor:
    def __init__(self, hotkey_text: str, fallback: str):
        self.fallback = fallback
        self.combo = parse_hotkey_text(hotkey_text, fallback)

    @property
    def text(self) -> str:
        return self.combo["text"]

    def update(self, hotkey_text: str):
        self.combo = parse_hotkey_text(hotkey_text, self.fallback)

    def is_pressed(self) -> bool:
        return is_hotkey_pressed(self.combo)
