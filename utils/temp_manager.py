import json
import logging
from copy import deepcopy
from pathlib import Path


logger = logging.getLogger(__name__)


DEFAULT_TEMP_DATA = {
    "drag_areas": {"plus": None, "minus": None, "time": None, "accept": None, "reject": None},
    "original_drag_areas": {"plus": None, "minus": None, "time": None, "accept": None, "reject": None},
    "temp_images": {"step1": None, "step2": None, "step3": None},
    "painted_images": {"plus": None, "minus": None, "time": None, "accept": None, "reject": None},
}


class TempManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if TempManager._instance is not None:
            raise RuntimeError("TempManager는 싱글톤 클래스입니다.")

        self.temp_dir = Path(__file__).parents[1] / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        self.temp_path = self.temp_dir / "tempdata.json"
        self._temp_data = self._load_temp_data()
        TempManager._instance = self

    def _load_temp_data(self):
        default_data = deepcopy(DEFAULT_TEMP_DATA)
        if not self.temp_path.exists():
            return default_data

        try:
            with open(self.temp_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("임시 데이터 로드 실패: %s", exc)
            return default_data

        for key, default_value in default_data.items():
            loaded_value = loaded.get(key)
            if not isinstance(default_value, dict):
                loaded.setdefault(key, default_value)
                continue
            if not isinstance(loaded_value, dict):
                loaded[key] = deepcopy(default_value)
                continue
            for sub_key, sub_default in default_value.items():
                loaded_value.setdefault(sub_key, sub_default)

        return loaded

    def save_temp_data(self) -> bool:
        try:
            with open(self.temp_path, "w", encoding="utf-8") as file:
                json.dump(self._temp_data, file, indent=4, ensure_ascii=False)
            return True
        except OSError as exc:
            logger.error("임시 데이터 저장 실패: %s", exc)
            return False

    @staticmethod
    def _serialize_rect(rect) -> dict | None:
        if rect is None:
            return None
        return {"x": rect.x(), "y": rect.y(), "width": rect.width(), "height": rect.height()}

    def save_drag_areas(self, drag_areas):
        current_areas = self._temp_data["drag_areas"]
        for label, rect in drag_areas.items():
            current_areas[label] = self._serialize_rect(rect)
        return self.save_temp_data()

    def get_drag_areas(self):
        return self._temp_data["drag_areas"]

    def save_original_drag_areas(self, original_drag_areas):
        current_orig = self._temp_data["original_drag_areas"]
        for label, rect in original_drag_areas.items():
            current_orig[label] = self._serialize_rect(rect)
        return self.save_temp_data()

    def get_original_drag_areas(self):
        return self._temp_data["original_drag_areas"]

    def save_temp_image(self, image, step):
        image_path = self.temp_dir / f"temp_step{step}.png"
        image.save(str(image_path))
        self._temp_data["temp_images"][f"step{step}"] = str(image_path)
        return self.save_temp_data()

    def get_temp_image(self, step):
        return self._temp_data["temp_images"][f"step{step}"]

    def save_painted_image(self, image, label):
        image_path = self.temp_dir / f"painted_{label}.png"
        image.save(str(image_path))
        self._temp_data["painted_images"][label] = str(image_path)
        return self.save_temp_data()

    def get_painted_image(self, label):
        return self._temp_data["painted_images"][label]

    def clear_temp_data(self):
        for path in self.temp_dir.glob("*.png"):
            path.unlink(missing_ok=True)
        if self.temp_path.exists():
            self.temp_path.unlink()
        self._temp_data = self._load_temp_data()
