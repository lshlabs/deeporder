from pathlib import Path
import copy
import json
import os
import shutil

from utils.path_manager import (
    data_path as managed_data_path,
    img_path as managed_img_path,
    make_relative_to_base,
    resolve_project_path,
)
from utils.temp_manager import TempManager


class DataManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if DataManager._instance is not None:
            raise Exception("DataManager는 싱글톤 클래스입니다.")

        self.data_path = managed_data_path()
        self.img_path = managed_img_path()
        self._data = self._load_data()
        DataManager._instance = self

    def _default_data(self):
        return {
            "macro_list": {},
            "settings_main": {
                "resolution": None,
                "custom": False,
            },
        }

    def _load_data(self):
        """data.json 로드 + 기본값 보정"""
        default_data = self._default_data()

        try:
            if self.data_path.exists():
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._normalize_loaded_data(data)
                return data

            self._normalize_loaded_data(default_data)
            return default_data
        except Exception as e:
            print(f"데이터 로드 중 오류: {e}")
            self._normalize_loaded_data(default_data)
            return default_data

    def _ensure_macro_defaults(self, macro: dict):
        macro.setdefault("program", None)

        settings = macro.setdefault("settings", {})
        settings.setdefault("matcher_mode", "ocr")
        settings.setdefault("max_retries", 10)
        settings.setdefault("retry_interval_sec", 0.5)
        settings.setdefault("template_timeout_sec", 10.0)

        macro.setdefault("actions", {})
        return macro

    def _normalize_loaded_data(self, data):
        data.setdefault("macro_list", {})
        data.setdefault("settings_main", {"resolution": None, "custom": False})

        for macro in data["macro_list"].values():
            self._ensure_macro_defaults(macro)

            for action in macro.get("actions", {}).values():
                if not isinstance(action, dict):
                    continue

                action.setdefault("enabled", True)
                action.setdefault("priority", False)

                if action.get("type") == "image" and action.get("image"):
                    resolved = resolve_project_path(action.get("image"))
                    if resolved is not None:
                        action["image"] = str(resolved)

    def _serialize_data(self):
        """저장 전에는 이미지 경로를 가능한 경우 프로젝트 상대경로로 바꾼다."""
        data_copy = copy.deepcopy(self._data)

        for macro in data_copy.get("macro_list", {}).values():
            for action in macro.get("actions", {}).values():
                if not isinstance(action, dict):
                    continue
                if action.get("type") != "image":
                    continue
                if not action.get("image"):
                    continue
                action["image"] = make_relative_to_base(action["image"])

        return data_copy

    def save_data(self):
        """현재 메모리의 데이터를 data.json에 저장"""
        try:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self._serialize_data(), f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"데이터 저장 중 오류: {e}")
            return False

    def _make_macro_folder(self, macro_key):
        macro_name = self._data["macro_list"][macro_key]["name"]
        macro_folder = self.img_path / macro_name
        macro_folder.mkdir(parents=True, exist_ok=True)
        return macro_folder

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
        """원본 이미지(step1)는 전체 이미지 크기를 좌표로 저장한다."""
        if not temp_image_path or not os.path.exists(temp_image_path):
            return None

        try:
            import cv2

            img = cv2.imread(str(temp_image_path))
            if img is None:
                return None
            h, w = img.shape[:2]
            return [0, 0, w, h]
        except Exception as e:
            print(f"원본 이미지 좌표 계산 실패: {e}")
            return None

    def _wizard_actions_common(self, macro_key, mapping, starting_action_number):
        """
        TempManager에 저장된 임시 이미지/영역 정보를 이용해서
        액션 데이터와 이미지 파일을 만든다.

        mapping 예시:
        {
            "step1": ("원본 이미지", "A1.png"),
            "minus": ("- 버튼 이미지", "A2.png"),
        }
        """
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

            if temp_image_path and os.path.exists(temp_image_path):
                target_image_path = macro_folder / filename
                shutil.copy2(temp_image_path, target_image_path)
            else:
                target_image_path = macro_folder / filename

            action_key = f"A{action_number}"
            actions[action_key] = self._make_image_action(
                action_name=action_name,
                action_number=action_number,
                image_path=target_image_path,
                coordinates=coordinates,
            )
            action_number += 1

        return actions

    def create_wizard_actions(self, macro_key):
        """새 매크로에 기본 6개 액션(원본/증감/시간/거절/수락) 생성"""
        self._ensure_macro_defaults(self._data["macro_list"][macro_key])

        mapping = {
            "step1": ("원본 이미지", "A1.png"),
            "minus": ("- 버튼 이미지", "A2.png"),
            "plus": ("+ 버튼 이미지", "A3.png"),
            "time": ("예상시간 이미지", "A4.png"),
            "reject": ("거절 버튼 이미지", "A5.png"),
            "accept": ("수락 버튼 이미지", "A6.png"),
        }

        actions = self._wizard_actions_common(
            macro_key,
            mapping,
            starting_action_number=1,
        )
        self._data["macro_list"][macro_key]["actions"] = actions
        return self.save_data()

    def add_wizard_actions(self, macro_key):
        """기존 매크로에 wizard 액션 6개를 추가"""
        macro = self._data["macro_list"][macro_key]
        self._ensure_macro_defaults(macro)
        actions = macro["actions"]

        last_action_number = max([a.get("number", 0) for a in actions.values()], default=0)

        mapping = {
            "step1": ("원본 이미지", f"A{last_action_number + 1}.png"),
            "minus": ("- 버튼 이미지", f"A{last_action_number + 2}.png"),
            "plus": ("+ 버튼 이미지", f"A{last_action_number + 3}.png"),
            "time": ("예상시간 이미지", f"A{last_action_number + 4}.png"),
            "reject": ("거절 버튼 이미지", f"A{last_action_number + 5}.png"),
            "accept": ("수락 버튼 이미지", f"A{last_action_number + 6}.png"),
        }

        new_actions = self._wizard_actions_common(
            macro_key,
            mapping,
            starting_action_number=last_action_number + 1,
        )
        actions.update(new_actions)
        return self.save_data()

    def create_delay_action(self, macro_key, delay_time):
        """딜레이 액션 추가"""
        try:
            actions = self._data["macro_list"][macro_key]["actions"]
            last_action_number = max([a.get("number", 0) for a in actions.values()], default=0)
            new_number = last_action_number + 1
            new_key = f"A{new_number}"

            actions[new_key] = {
                "name": f"딜레이 {delay_time}초",
                "type": "delay",
                "number": new_number,
                "value": float(delay_time),
                "priority": False,
                "enabled": True,
            }
            return self.save_data()
        except Exception as e:
            print(f"딜레이 액션 생성 중 오류: {e}")
            return False

    def copy_macro(self, original_macro_key, new_name):
        """매크로 복제 + 이미지 폴더 복제"""
        try:
            macro_keys = self._data["macro_list"].keys()
            next_num = 1
            while f"M{next_num}" in macro_keys:
                next_num += 1
            new_macro_key = f"M{next_num}"

            original_macro = self._data["macro_list"][original_macro_key]
            new_macro = {
                "name": new_name,
                "program": original_macro.get("program"),
                "settings": original_macro.get("settings", {}).copy(),
                "actions": {},
            }

            for action_key, action_data in original_macro.get("actions", {}).items():
                if isinstance(action_data, dict):
                    new_macro["actions"][action_key] = action_data.copy()
                else:
                    new_macro["actions"][action_key] = action_data

            original_folder = self.img_path / original_macro["name"]
            new_folder = self.img_path / new_name

            if original_folder.exists():
                if new_folder.exists():
                    shutil.rmtree(new_folder)
                shutil.copytree(original_folder, new_folder)

                # 복제된 액션의 이미지 경로를 새 폴더 기준으로 수정
                for action in new_macro["actions"].values():
                    if not isinstance(action, dict):
                        continue
                    if action.get("type") != "image":
                        continue
                    if not action.get("image"):
                        continue
                    old_path = Path(action["image"])
                    action["image"] = str((new_folder / old_path.name).resolve())

            self._data["macro_list"][new_macro_key] = new_macro
            self.save_data()
            return new_macro_key

        except Exception as e:
            print(f"매크로 복제 중 오류: {e}")
            return None
