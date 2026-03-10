from __future__ import annotations

import ctypes
import os

from utils.types import HotkeyComboData


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000


def parse_hotkey_text(text: str, fallback: str) -> HotkeyComboData:
    alias_map: dict[str, tuple[str, list[int]]] = {
        "CTRL": ("Ctrl", [0x11]),
        "CONTROL": ("Ctrl", [0x11]),
        "ALT": ("Alt", [0x12]),
        "SHIFT": ("Shift", [0x10]),
        "META": ("Meta", [0x5B, 0x5C]),
        "WIN": ("Meta", [0x5B, 0x5C]),
        "WINDOWS": ("Meta", [0x5B, 0x5C]),
    }
    special_keys: dict[str, tuple[str, int]] = {
        "TAB": ("Tab", 0x09),
        "SPACE": ("Space", 0x20),
        "ENTER": ("Enter", 0x0D),
        "RETURN": ("Enter", 0x0D),
        "ESC": ("Esc", 0x1B),
        "ESCAPE": ("Esc", 0x1B),
    }

    tokens = [part.strip() for part in str(text or "").split("+") if part.strip()]
    if not tokens:
        tokens = [fallback]

    display_parts: list[str] = []
    modifiers: list[list[int]] = []
    primary: list[int] | None = None

    for token in tokens:
        normalized = token.upper()
        if normalized in alias_map:
            label, virtual_keys = alias_map[normalized]
            if label not in display_parts:
                display_parts.append(label)
                modifiers.append(virtual_keys)
            continue

        if normalized.startswith("F") and normalized[1:].isdigit():
            number = int(normalized[1:])
            if 1 <= number <= 24:
                display_parts.append(f"F{number}")
                primary = [0x6F + number]
                continue

        if normalized in special_keys:
            label, virtual_key = special_keys[normalized]
            display_parts.append(label)
            primary = [virtual_key]
            continue

        if len(normalized) == 1:
            code = ord(normalized)
            if 48 <= code <= 57 or 65 <= code <= 90:
                display_parts.append(normalized)
                primary = [code]

    if primary is None:
        if str(text or "").strip().upper() == str(fallback or "").strip().upper():
            return {"text": "F12", "modifiers": [], "primary": [0x7B]}
        return parse_hotkey_text(fallback, fallback)

    return {
        "text": "+".join(display_parts),
        "modifiers": modifiers,
        "primary": primary,
    }


def is_hotkey_pressed(combo: HotkeyComboData) -> bool:
    if os.name != "nt":
        return False

    user32 = getattr(ctypes, "windll", None)
    if user32 is None or not hasattr(user32, "user32"):
        return False

    try:
        keyboard_api = user32.user32
        for group in combo["modifiers"]:
            if not any(keyboard_api.GetAsyncKeyState(virtual_key) & 0x8000 for virtual_key in group):
                return False
        return any(keyboard_api.GetAsyncKeyState(virtual_key) & 0x8000 for virtual_key in combo["primary"])
    except OSError:
        return False


def parse_hotkey_for_register(text: str, fallback: str) -> tuple[int, int]:
    combo = parse_hotkey_text(text, fallback)
    modifier_flags = 0

    for group in combo["modifiers"]:
        group_set = set(group)
        if group_set == {0x11}:
            modifier_flags |= MOD_CONTROL
        elif group_set == {0x12}:
            modifier_flags |= MOD_ALT
        elif group_set == {0x10}:
            modifier_flags |= MOD_SHIFT
        elif group_set == {0x5B, 0x5C}:
            modifier_flags |= MOD_WIN

    return modifier_flags | MOD_NOREPEAT, combo["primary"][0]
