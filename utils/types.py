from __future__ import annotations

from typing import Literal, TypedDict


class RectData(TypedDict):
    x: int
    y: int
    width: int
    height: int


class ColorData(TypedDict):
    r: int
    g: int
    b: int


class TriggerData(TypedDict, total=False):
    trigger_type: Literal["pixel_color", "text_roi"]
    screen_rect: RectData
    expected_color: ColorData | None
    color_tolerance: int
    expected_text: str | None
    text_match_mode: Literal["contains", "exact"]
    ocr_confidence_min: float
    enabled: bool


class StepData(TypedDict, total=False):
    step_type: Literal["item", "delay", "note"]
    item_id: str | None
    click_count: int
    order: int
    enabled: bool
    delay_sec: float | None
    note_text: str | None


class PresetData(TypedDict):
    id: str
    name: str
    preset_trigger: TriggerData | None
    steps: list[StepData]


class ItemData(TypedDict):
    id: str
    name: str
    item_type: Literal["button", "text"]
    screen_rect: RectData | None
    capture_rect: RectData | None
    preview_image: str | None
    enabled: bool


class MacroSettingsData(TypedDict):
    matcher_mode: str
    max_retries: int
    retry_interval_sec: float
    template_timeout_sec: float
    repeat_count: int
    repeat_delay_sec: float


class MacroData(TypedDict, total=False):
    name: str
    program: str | None
    capture_source_image: str | None
    settings: MacroSettingsData
    actions: dict[str, dict]
    items: dict[str, ItemData]
    presets: dict[str, PresetData]
    default_preset_id: str | None
    macro_trigger: TriggerData | None
    schema_version: int


class AppSettingsData(TypedDict):
    resolution: str | None
    custom: bool
    setup_completed: bool
    capture_hotkey: str
    run_hotkey: str
    stop_hotkey: str
    expected_resolution: str | None
    dpi_scale_locked: bool


class HotkeyComboData(TypedDict):
    text: str
    modifiers: list[list[int]]
    primary: list[int]
