from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any

from core_functions.image_matcher import ImageMatcher as TemplateImageMatcher
from image_matcher_easyocr import ImageMatcherEasyOCR


class BaseMatcher(ABC):
    @abstractmethod
    def find_template(self, template_id: str):
        raise NotImplementedError


class TemplateMatcherAdapter(BaseMatcher):
    def __init__(self, threshold: float = 0.7):
        self.matcher = TemplateImageMatcher(threshold=threshold)

    def find_template(self, template_id: str):
        return self.matcher.find_template(template_id)


class OCRMatcherAdapter(BaseMatcher):
    def __init__(self, threshold: float = 0.7):
        self.matcher = ImageMatcherEasyOCR(threshold=threshold)

    def find_template(self, template_id: str):
        return self.matcher.find_template(template_id)


class VisionEngine:
    """
    Hybrid runtime facade:
    - detection is handled by selected matcher (ocr/template)
    - template metadata/coordinates are always served from template matcher
    """

    def __init__(self, mode: str = "ocr", threshold: float = 0.7):
        self.mode = mode if mode in {"ocr", "template"} else "ocr"
        self.threshold = threshold
        self.template_adapter = TemplateMatcherAdapter(threshold=threshold)
        self.ocr_adapter = OCRMatcherAdapter(threshold=threshold)

    @property
    def metadata_matcher(self):
        return self.template_adapter.matcher

    @property
    def active_matcher(self):
        return self.ocr_adapter.matcher if self.mode == "ocr" else self.template_adapter.matcher

    @property
    def template_paths(self):
        return self.metadata_matcher.template_paths

    @property
    def template_sizes(self):
        return self.metadata_matcher.template_sizes

    @property
    def template_actions(self):
        return self.metadata_matcher.template_actions

    def reload_templates(self) -> None:
        self.template_adapter = TemplateMatcherAdapter(threshold=self.threshold)

    def find_template(self, template_id: str):
        # Runtime macros pass IDs like "M1_A1". These require template-origin coordinates
        # for downstream action offset math, so keep template matcher as the source of truth.
        if re.match(r"^M\d+_A\d+$", str(template_id)):
            return self.template_adapter.find_template(template_id)
        return self.active_matcher.find_template(template_id)

    def load_template(self, template_id: str):
        return self.metadata_matcher.load_template(template_id)

    def get_scaled_action_coordinates(self, template_id: str, action_id: str, template_location: Any, scale_info: Any):
        return self.metadata_matcher.get_scaled_action_coordinates(template_id, action_id, template_location, scale_info)

    def get_action_center(self, template_id: str, action_id: str, template_location: Any, scale_info: Any):
        return self.metadata_matcher.get_action_center(template_id, action_id, template_location, scale_info)
