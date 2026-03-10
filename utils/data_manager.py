from __future__ import annotations

import copy
import json
import logging
import os
import shutil
from pathlib import Path

from utils.path_manager import data_path as managed_data_path
from utils.path_manager import img_path as managed_img_path
from utils.path_manager import make_relative_to_base, resolve_project_path
from utils.temp_manager import TempManager
from utils.types import AppSettingsData, MacroData


logger = logging.getLogger(__name__)


class DataManager:
    _instance = None
    SCHEMA_VERSION = 2
    MAIN_SETTINGS_DEFAULTS = {
        "resolution": None,
        "custom": False,
        "setup_completed": False,
        "capture_hotkey": "F1",
        "run_hotkey": "F11",
        "stop_hotkey": "F12",
        "expected_resolution": None,
        "dpi_scale_locked": False,
    }
    MACRO_SETTINGS_DEFAULTS = {
        "matcher_mode": "ocr",
        "max_retries": 10,
        "retry_interval_sec": 0.5,
        "template_timeout_sec": 10.0,
        "repeat_count": 1,
        "repeat_delay_sec": 0.5,
    }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if DataManager._instance is not None:
            raise RuntimeError("DataManager는 싱글톤 클래스입니다.")
        self.data_path = managed_data_path()
        self.img_path = managed_img_path()
        self._data = self._load_data()
        DataManager._instance = self

    def _default_data(self):
        return {
            "macro_list": {},
            "settings_main": copy.deepcopy(self.MAIN_SETTINGS_DEFAULTS),
        }

    def _load_data(self):
        default_data = self._default_data()
        if not self.data_path.exists():
            self._normalize_loaded_data(default_data)
            return default_data

        try:
            with open(self.data_path, "r", encoding="utf-8") as file:
                loaded_data = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("데이터 로드 실패: %s", exc)
            self._normalize_loaded_data(default_data)
            return default_data

        self._normalize_loaded_data(loaded_data)
        return loaded_data

    def _ensure_macro_defaults(self, macro: dict):
        macro.setdefault("program", None)
        macro.setdefault("capture_source_image", None)
        settings = macro.setdefault("settings", {})
        for key, value in self.MACRO_SETTINGS_DEFAULTS.items():
            settings.setdefault(key, value)
        macro.setdefault("actions", {})
        macro.setdefault("items", {})
        macro.setdefault("presets", {})
        macro.setdefault("default_preset_id", None)
        macro.setdefault("macro_trigger", None)
        macro.setdefault("schema_version", self.SCHEMA_VERSION)

    def _new_macro_payload(self, macro_name: str) -> dict:
        return {
            "name": macro_name,
            "program": None,
            "capture_source_image": None,
            "settings": copy.deepcopy(self.MACRO_SETTINGS_DEFAULTS),
            "actions": {},
            "items": {},
            "presets": {"P1": {"id": "P1", "name": macro_name, "preset_trigger": None, "steps": []}},
            "default_preset_id": "P1",
            "macro_trigger": None,
            "schema_version": self.SCHEMA_VERSION,
        }

    def _copy_if_file_exists(self, source_path: str | None, target_path: Path) -> str | None:
        if not source_path or not os.path.exists(source_path):
            return None
        shutil.copy2(source_path, target_path)
        return str(target_path.resolve())

    def _build_item_from_region(self, region: dict, item_id: str, fallback_name: str) -> dict:
        item_type = "text" if region.get("item_type") == "text" else "button"
        screen_rect = self._normalize_rect(region.get("screen_rect"))
        return {
            "id": item_id,
            "name": region.get("name") or fallback_name,
            "item_type": item_type,
            "screen_rect": screen_rect,
            "capture_rect": self._normalize_rect(region.get("capture_rect")) or screen_rect,
            "preview_image": None,
            "enabled": True,
        }

    @staticmethod
    def _new_item_step(item_id: str, order: int, click_count: int = 1) -> dict:
        return {
            "step_type": "item",
            "item_id": item_id,
            "click_count": int(click_count or 1),
            "order": order,
            "enabled": True,
            "delay_sec": None,
        }

    def _normalize_rect(self, rect):
        if not isinstance(rect, dict):
            return None
        if not all(key in rect for key in ("x", "y", "width", "height")):
            return None
        return {
            "x": int(rect["x"]),
            "y": int(rect["y"]),
            "width": int(rect["width"]),
            "height": int(rect["height"]),
        }

    def _normalize_trigger(self, trigger):
        if not isinstance(trigger, dict):
            return None
        merged = {
            "trigger_type": trigger.get("trigger_type"),
            "screen_rect": self._normalize_rect(trigger.get("screen_rect")),
            "expected_color": trigger.get("expected_color"),
            "color_tolerance": int(trigger.get("color_tolerance", 10) or 10),
            "expected_text": trigger.get("expected_text"),
            "text_match_mode": trigger.get("text_match_mode", "contains"),
            "ocr_confidence_min": float(trigger.get("ocr_confidence_min", 0.6) or 0.6),
            "enabled": bool(trigger.get("enabled", False)),
        }
        return merged

    def _normalize_step(self, step: dict, order: int):
        step_type = step.get("step_type", "item")
        return {
            "step_type": step_type,
            "item_id": step.get("item_id"),
            "click_count": int(step.get("click_count", 1) or 1),
            "order": int(step.get("order", order) or order),
            "enabled": bool(step.get("enabled", True)),
            "delay_sec": float(step.get("delay_sec", 0.0) or 0.0) if step_type == "delay" else None,
            "note_text": str(step.get("note_text") or "메모") if step_type == "note" else None,
        }

    def _sort_steps(self, macro: dict, steps: list[dict]):
        def step_key(step):
            if step.get("step_type") in {"delay", "note"}:
                return (0, int(step.get("order", 0)))
            item = macro.get("items", {}).get(step.get("item_id"), {})
            return (1 if item.get("item_type") == "text" else 0, int(step.get("order", 0)))

        sorted_steps = sorted(steps, key=step_key)
        for index, step in enumerate(sorted_steps, start=1):
            step["order"] = index
        return sorted_steps

    def _normalize_loaded_data(self, data):
        defaults = self._default_data()
        settings = data.setdefault("settings_main", {})
        for key, value in defaults["settings_main"].items():
            settings.setdefault(key, value)

        data.setdefault("macro_list", {})
        for macro in data["macro_list"].values():
            self._ensure_macro_defaults(macro)
            self._migrate_legacy_macro(macro)
            for action in macro.get("actions", {}).values():
                if isinstance(action, dict) and action.get("type") == "image" and action.get("image"):
                    resolved = resolve_project_path(action["image"])
                    if resolved is not None:
                        action["image"] = str(resolved)
            if macro.get("capture_source_image"):
                resolved = resolve_project_path(macro["capture_source_image"])
                if resolved is not None:
                    macro["capture_source_image"] = str(resolved)
            for item in macro.get("items", {}).values():
                if not isinstance(item, dict):
                    continue
                item.setdefault("enabled", True)
                item.setdefault("item_type", "button")
                item["screen_rect"] = self._normalize_rect(item.get("screen_rect"))
                item["capture_rect"] = self._normalize_rect(item.get("capture_rect")) or item["screen_rect"]
                if item.get("preview_image"):
                    resolved = resolve_project_path(item["preview_image"])
                    if resolved is not None:
                        item["preview_image"] = str(resolved)
            if macro.get("macro_trigger"):
                macro["macro_trigger"] = self._normalize_trigger(macro["macro_trigger"])
            for preset_id, preset in macro.get("presets", {}).items():
                preset.setdefault("id", preset_id)
                preset.setdefault("name", preset_id)
                preset["preset_trigger"] = self._normalize_trigger(preset.get("preset_trigger"))
                normalized = []
                for index, step in enumerate(preset.get("steps", []), start=1):
                    if isinstance(step, dict):
                        normalized.append(self._normalize_step(step, index))
                preset["steps"] = self._sort_steps(macro, normalized)
            if not macro.get("default_preset_id") and macro.get("presets"):
                macro["default_preset_id"] = next(iter(macro["presets"]))
            macro["schema_version"] = self.SCHEMA_VERSION

    def _serialize_data(self):
        data_copy = copy.deepcopy(self._data)
        for macro in data_copy.get("macro_list", {}).values():
            for action in macro.get("actions", {}).values():
                if isinstance(action, dict) and action.get("type") == "image" and action.get("image"):
                    action["image"] = make_relative_to_base(action["image"])
            if macro.get("capture_source_image"):
                macro["capture_source_image"] = make_relative_to_base(macro["capture_source_image"])
            for item in macro.get("items", {}).values():
                if isinstance(item, dict) and item.get("preview_image"):
                    item["preview_image"] = make_relative_to_base(item["preview_image"])
        return data_copy

    def save_data(self):
        try:
            with open(self.data_path, "w", encoding="utf-8") as file:
                json.dump(self._serialize_data(), file, indent=4, ensure_ascii=False)
            return True
        except OSError as exc:
            logger.error("데이터 저장 실패: %s", exc)
            return False

    def _next_macro_key(self):
        index = 1
        while f"M{index}" in self._data["macro_list"]:
            index += 1
        return f"M{index}"

    def _next_item_id(self, macro: dict):
        index = 1
        while f"I{index}" in macro["items"]:
            index += 1
        return f"I{index}"

    def _next_preset_id(self, macro: dict):
        index = 1
        while f"P{index}" in macro["presets"]:
            index += 1
        return f"P{index}"

    def get_macro(self, macro_key: str):
        macro = self._data["macro_list"].get(macro_key)
        if macro:
            self._migrate_legacy_macro(macro)
        return macro

    def restore_macro(self, macro_key: str, macro_snapshot: MacroData) -> bool:
        self._data["macro_list"][macro_key] = copy.deepcopy(macro_snapshot)
        return self.save_data()

    def get_macro_list(self) -> dict[str, MacroData]:
        for macro in self._data["macro_list"].values():
            if isinstance(macro, dict):
                self._migrate_legacy_macro(macro)
        return self._data["macro_list"]

    def get_next_item_id(self, macro_key: str) -> str:
        macro = self.get_macro(macro_key)
        if not macro:
            return "I1"
        return self._next_item_id(macro)

    def delete_macro(self, macro_key: str) -> bool:
        macro = self._data["macro_list"].get(macro_key)
        if not macro:
            return False

        macro_name = str(macro.get("name") or "")
        if macro_name:
            macro_folder = self.img_path / macro_name
            if macro_folder.exists():
                shutil.rmtree(macro_folder)

        del self._data["macro_list"][macro_key]
        return self.save_data()

    def get_settings(self) -> AppSettingsData:
        return self._data["settings_main"]

    def update_settings(self, **updates) -> bool:
        settings = self._data["settings_main"]
        settings.update(updates)
        return self.save_data()

    def is_setup_completed(self) -> bool:
        return bool(self.get_settings().get("setup_completed"))

    def get_capture_hotkey(self) -> str:
        return str(self.get_settings().get("capture_hotkey") or "F1")

    def get_run_hotkey(self) -> str:
        return str(self.get_settings().get("run_hotkey") or "F11")

    def get_stop_hotkey(self) -> str:
        return str(self.get_settings().get("stop_hotkey") or "F12")

    def get_active_preset(self, macro_key: str, preset_id: str | None = None):
        macro = self.get_macro(macro_key)
        if not macro:
            return None
        current_id = preset_id or macro.get("default_preset_id")
        return macro.get("presets", {}).get(current_id)

    def sort_preset_steps(self, macro_key: str, preset_id: str):
        macro = self.get_macro(macro_key)
        preset = self.get_active_preset(macro_key, preset_id)
        if not macro or not preset:
            return []
        preset["steps"] = self._sort_steps(macro, preset.get("steps", []))
        return preset["steps"]

    def validate_macro_configuration(self, macro_key: str):
        macro = self.get_macro(macro_key)
        if not macro:
            return False, "매크로 정보를 찾을 수 없습니다."
        if not macro.get("items"):
            return False, "항목이 하나 이상 필요합니다."
        default_id = macro.get("default_preset_id")
        for preset_id, preset in macro.get("presets", {}).items():
            executable_steps = [
                step
                for step in preset.get("steps", [])
                if step.get("enabled", True) and step.get("step_type") != "note"
            ]
            if not executable_steps:
                return False, f"{preset.get('name', preset_id)}에 활성 단계가 없습니다."
            if preset_id != default_id and not preset.get("preset_trigger"):
                return False, f"{preset.get('name', preset_id)}의 실행 트리거가 필요합니다."
        return True, ""

    def _make_macro_folder(self, macro_key):
        macro_name = self._data["macro_list"][macro_key]["name"]
        macro_folder = self.img_path / macro_name
        macro_folder.mkdir(parents=True, exist_ok=True)
        return macro_folder

    def _ensure_unique_macro_name(self, name: str):
        existing = {
            macro.get("name")
            for macro in self._data.get("macro_list", {}).values()
            if isinstance(macro, dict)
        }
        if name not in existing:
            return name
        index = 1
        while True:
            candidate = f"{name} ({index})"
            if candidate not in existing:
                return candidate
            index += 1

    def create_macro_from_capture(self, macro_name: str, capture_regions: list[dict], source_image_path: str | None = None):
        safe_name = self._ensure_unique_macro_name(macro_name.strip() or "새 매크로")
        macro_key = self._next_macro_key()
        self._data["macro_list"][macro_key] = self._new_macro_payload(safe_name)
        macro = self._data["macro_list"][macro_key]
        macro_folder = self._make_macro_folder(macro_key)
        macro["capture_source_image"] = self._copy_if_file_exists(source_image_path, macro_folder / "capture_source.png")

        for region in capture_regions:
            item_id = self._next_item_id(macro)
            fallback_name = f"{'텍스트' if region.get('item_type') == 'text' else '버튼'} {len(macro['items']) + 1}"
            item = self._build_item_from_region(region, item_id, fallback_name)
            item["preview_image"] = self._copy_if_file_exists(region.get("preview_image"), macro_folder / f"{item_id}.png")
            macro["items"][item_id] = item
            macro["presets"]["P1"]["steps"].append(
                self._new_item_step(
                    item_id=item_id,
                    order=len(macro["presets"]["P1"]["steps"]) + 1,
                    click_count=int(region.get("click_count", 1) or 1),
                )
            )

        self.sort_preset_steps(macro_key, "P1")
        self.save_data()
        return macro_key

    def append_items_from_capture(self, macro_key: str, capture_regions: list[dict], preset_id: str | None = None):
        macro = self.get_macro(macro_key)
        preset = self.get_active_preset(macro_key, preset_id)
        if not macro or not preset:
            return False

        macro_folder = self._make_macro_folder(macro_key)
        for region in capture_regions:
            item_id = self._next_item_id(macro)
            item = self._build_item_from_region(region, item_id, f"항목 {len(macro['items']) + 1}")
            item["preview_image"] = self._copy_if_file_exists(region.get("preview_image"), macro_folder / f"{item_id}.png")
            macro["items"][item_id] = item
            preset["steps"].append(self._new_item_step(item_id=item_id, order=len(preset["steps"]) + 1))

        self.sort_preset_steps(macro_key, preset["id"])
        return self.save_data()

    def add_preset(self, macro_key: str, source_preset_id: str | None = None, name: str | None = None):
        macro = self.get_macro(macro_key)
        if not macro:
            return None
        preset_id = self._next_preset_id(macro)
        source = self.get_active_preset(macro_key, source_preset_id or macro.get("default_preset_id"))
        macro["presets"][preset_id] = {
            "id": preset_id,
            "name": name or f"프리셋 {len(macro['presets']) + 1}",
            "preset_trigger": None,
            "steps": copy.deepcopy(source.get("steps", []) if source else []),
        }
        self.sort_preset_steps(macro_key, preset_id)
        self.save_data()
        return preset_id

    def rename_preset(self, macro_key: str, preset_id: str, name: str):
        preset = self.get_active_preset(macro_key, preset_id)
        if not preset:
            return False
        preset["name"] = name.strip() or preset["name"]
        return self.save_data()

    def delete_preset(self, macro_key: str, preset_id: str):
        macro = self.get_macro(macro_key)
        if not macro or preset_id == macro.get("default_preset_id") or preset_id not in macro.get("presets", {}):
            return False
        del macro["presets"][preset_id]
        return self.save_data()

    def set_item_type(self, macro_key: str, item_id: str, item_type: str):
        macro = self.get_macro(macro_key)
        if not macro or item_id not in macro["items"]:
            return False
        macro["items"][item_id]["item_type"] = "text" if item_type == "text" else "button"
        for preset_id in macro.get("presets", {}):
            self.sort_preset_steps(macro_key, preset_id)
        return self.save_data()

    def update_item(self, macro_key: str, item_id: str, **updates):
        macro = self.get_macro(macro_key)
        if not macro or item_id not in macro["items"]:
            return False
        macro["items"][item_id].update(updates)
        return self.save_data()

    def update_step(self, macro_key: str, preset_id: str, step_index: int, **updates):
        preset = self.get_active_preset(macro_key, preset_id)
        if not preset or step_index < 0 or step_index >= len(preset.get("steps", [])):
            return False
        preset["steps"][step_index].update(updates)
        self.sort_preset_steps(macro_key, preset_id)
        return self.save_data()

    def add_delay_step(self, macro_key: str, preset_id: str, delay_time: float):
        preset = self.get_active_preset(macro_key, preset_id)
        if not preset:
            return False
        preset["steps"].append(
            {
                "step_type": "delay",
                "item_id": None,
                "click_count": 1,
                "order": len(preset["steps"]) + 1,
                "enabled": True,
                "delay_sec": float(delay_time),
            }
        )
        self.sort_preset_steps(macro_key, preset_id)
        return self.save_data()

    def add_note_step(self, macro_key: str, preset_id: str, note_text: str):
        preset = self.get_active_preset(macro_key, preset_id)
        if not preset:
            return False
        preset["steps"].append(
            {
                "step_type": "note",
                "item_id": None,
                "click_count": 1,
                "order": len(preset["steps"]) + 1,
                "enabled": True,
                "delay_sec": None,
                "note_text": note_text.strip() or "메모",
            }
        )
        self.sort_preset_steps(macro_key, preset_id)
        return self.save_data()

    def move_step(self, macro_key: str, preset_id: str, step_index: int, direction: int):
        preset = self.get_active_preset(macro_key, preset_id)
        macro = self.get_macro(macro_key)
        if not preset or not macro:
            return False
        steps = self.sort_preset_steps(macro_key, preset_id)
        if step_index < 0 or step_index >= len(steps):
            return False
        target_index = step_index + direction
        if target_index < 0 or target_index >= len(steps):
            return False

        step = steps[step_index]
        target = steps[target_index]
        if step.get("step_type") == "item":
            item = macro.get("items", {}).get(step.get("item_id"), {})
            if item.get("item_type") == "text":
                return False
        if target.get("step_type") == "item":
            target_item = macro.get("items", {}).get(target.get("item_id"), {})
            if target_item.get("item_type") == "text":
                return False

        steps[step_index], steps[target_index] = steps[target_index], steps[step_index]
        for index, current in enumerate(steps, start=1):
            current["order"] = index
        return self.save_data()

    def set_macro_trigger(self, macro_key: str, trigger: dict | None):
        macro = self.get_macro(macro_key)
        if not macro:
            return False
        macro["macro_trigger"] = self._normalize_trigger(trigger)
        return self.save_data()

    def set_preset_trigger(self, macro_key: str, preset_id: str, trigger: dict | None):
        preset = self.get_active_preset(macro_key, preset_id)
        if not preset:
            return False
        preset["preset_trigger"] = self._normalize_trigger(trigger)
        return self.save_data()

    def _make_image_action(self, action_name, action_number, image_path, coordinates=None):
        return {
            "name": action_name,
            "type": "image",
            "number": action_number,
            "image": str(Path(image_path).resolve()),
            "priority": False,
            "enabled": True,
            "coordinates": coordinates,
        }

    def _get_step1_coordinates(self, temp_image_path):
        if not temp_image_path or not os.path.exists(temp_image_path):
            return None
        try:
            import cv2

            img = cv2.imread(str(temp_image_path))
            if img is None:
                return None
            height, width = img.shape[:2]
            return [0, 0, width, height]
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.error("원본 이미지 좌표 계산 실패: %s", exc)
            return None

    def _wizard_actions_common(self, macro_key, mapping, starting_action_number):
        temp_manager = TempManager.get_instance()
        macro_folder = self._make_macro_folder(macro_key)
        original_drag_areas = temp_manager.get_original_drag_areas()
        actions = {}
        action_number = starting_action_number

        for temp_label, (action_name, filename) in mapping.items():
            coordinates = None
            if temp_label == "step1":
                temp_image_path = temp_manager.get_temp_image(1)
                coordinates = self._get_step1_coordinates(temp_image_path)
            else:
                temp_image_path = temp_manager.get_painted_image(temp_label)
                drag_area = original_drag_areas.get(temp_label)
                if drag_area:
                    coordinates = [
                        drag_area["x"],
                        drag_area["y"],
                        drag_area["width"],
                        drag_area["height"],
                    ]

            target_image_path = macro_folder / filename
            if temp_image_path and os.path.exists(temp_image_path):
                shutil.copy2(temp_image_path, target_image_path)
            action_key = f"A{action_number}"
            actions[action_key] = self._make_image_action(action_name, action_number, target_image_path, coordinates)
            action_number += 1
        return actions

    def create_wizard_actions(self, macro_key):
        self._ensure_macro_defaults(self._data["macro_list"][macro_key])
        mapping = {
            "step1": ("원본 이미지", "A1.png"),
            "minus": ("- 버튼 이미지", "A2.png"),
            "plus": ("+ 버튼 이미지", "A3.png"),
            "time": ("예상시간 이미지", "A4.png"),
            "reject": ("거절 버튼 이미지", "A5.png"),
            "accept": ("수락 버튼 이미지", "A6.png"),
        }
        self._data["macro_list"][macro_key]["actions"] = self._wizard_actions_common(macro_key, mapping, 1)
        self._migrate_legacy_macro(self._data["macro_list"][macro_key], force=True)
        return self.save_data()

    def add_wizard_actions(self, macro_key):
        macro = self._data["macro_list"][macro_key]
        actions = macro.setdefault("actions", {})
        last_action_number = max([a.get("number", 0) for a in actions.values()], default=0)
        mapping = {
            "step1": ("원본 이미지", f"A{last_action_number + 1}.png"),
            "minus": ("- 버튼 이미지", f"A{last_action_number + 2}.png"),
            "plus": ("+ 버튼 이미지", f"A{last_action_number + 3}.png"),
            "time": ("예상시간 이미지", f"A{last_action_number + 4}.png"),
            "reject": ("거절 버튼 이미지", f"A{last_action_number + 5}.png"),
            "accept": ("수락 버튼 이미지", f"A{last_action_number + 6}.png"),
        }
        actions.update(self._wizard_actions_common(macro_key, mapping, last_action_number + 1))
        self._migrate_legacy_macro(macro, force=True)
        return self.save_data()

    def create_delay_action(self, macro_key, delay_time):
        macro = self.get_macro(macro_key)
        preset_id = macro.get("default_preset_id") if macro else None
        if preset_id:
            return self.add_delay_step(macro_key, preset_id, float(delay_time))
        return False

    def copy_macro(self, original_macro_key, new_name):
        try:
            original_macro = self.get_macro(original_macro_key)
            if not original_macro:
                return None
            new_macro_key = self._next_macro_key()
            new_name = self._ensure_unique_macro_name(new_name)
            new_macro = copy.deepcopy(original_macro)
            new_macro["name"] = new_name

            original_folder = self.img_path / original_macro["name"]
            new_folder = self.img_path / new_name
            if original_folder.exists():
                if new_folder.exists():
                    shutil.rmtree(new_folder)
                shutil.copytree(original_folder, new_folder)

                if new_macro.get("capture_source_image"):
                    new_macro["capture_source_image"] = str((new_folder / Path(new_macro["capture_source_image"]).name).resolve())
                for action in new_macro.get("actions", {}).values():
                    if isinstance(action, dict) and action.get("type") == "image" and action.get("image"):
                        action["image"] = str((new_folder / Path(action["image"]).name).resolve())
                for item in new_macro.get("items", {}).values():
                    if isinstance(item, dict) and item.get("preview_image"):
                        item["preview_image"] = str((new_folder / Path(item["preview_image"]).name).resolve())

            self._data["macro_list"][new_macro_key] = new_macro
            self.save_data()
            return new_macro_key
        except (OSError, shutil.Error, KeyError, ValueError) as exc:
            logger.error("매크로 복제 실패: %s", exc)
            return None

    def _infer_item_type_from_name(self, name: str):
        lowered = str(name).lower()
        if any(token in lowered for token in ("시간", "텍스트", "text")):
            return "text"
        return "button"

    def _migrate_legacy_macro(self, macro: dict, force: bool = False):
        if macro.get("items") and macro.get("presets") and not force:
            return

        if not macro.get("actions"):
            if not macro.get("presets"):
                preset_id = "P1"
                macro["presets"][preset_id] = {"id": preset_id, "name": macro.get("name", "기본 프리셋"), "preset_trigger": None, "steps": []}
                macro["default_preset_id"] = preset_id
            return

        items = {}
        steps = []
        sorted_actions = sorted(
            macro.get("actions", {}).items(),
            key=lambda x: x[1].get("number", 0) if isinstance(x[1], dict) else 0,
        )
        for _, action in sorted_actions:
            if not isinstance(action, dict):
                continue
            if action.get("type") == "delay":
                steps.append(
                    {
                        "step_type": "delay",
                        "item_id": None,
                        "click_count": 1,
                        "order": len(steps) + 1,
                        "enabled": bool(action.get("enabled", True)),
                        "delay_sec": float(action.get("value", 0.0) or 0.0),
                    }
                )
                continue
            if action.get("type") != "image":
                continue

            item_id = f"I{len(items) + 1}"
            rect = None
            coords = action.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 4:
                rect = {"x": int(coords[0]), "y": int(coords[1]), "width": int(coords[2]), "height": int(coords[3])}
            items[item_id] = {
                "id": item_id,
                "name": action.get("name", item_id),
                "item_type": self._infer_item_type_from_name(action.get("name", "")),
                "screen_rect": copy.deepcopy(rect),
                "capture_rect": copy.deepcopy(rect),
                "preview_image": action.get("image"),
                "enabled": bool(action.get("enabled", True)),
            }
            steps.append(
                {
                    "step_type": "item",
                    "item_id": item_id,
                    "click_count": 1,
                    "order": len(steps) + 1,
                    "enabled": bool(action.get("enabled", True)),
                    "delay_sec": None,
                }
            )

        macro["items"] = items
        macro["presets"] = {
            "P1": {"id": "P1", "name": macro.get("name", "기본 프리셋"), "preset_trigger": None, "steps": steps}
        }
        macro["default_preset_id"] = "P1"
        macro["schema_version"] = self.SCHEMA_VERSION
        macro["presets"]["P1"]["steps"] = self._sort_steps(macro, macro["presets"]["P1"]["steps"])
