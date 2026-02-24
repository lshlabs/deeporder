from __future__ import annotations

import re
from typing import Any

from core_functions.image_matcher import ImageMatcher as TemplateImageMatcher
from image_matcher_easyocr import ImageMatcherEasyOCR


class VisionEngine:
    """
    MacroRunner에서 매칭 방식을 바꿔 쓸 수 있게 해주는 래퍼 클래스.

    - template 매칭(OpenCV)
    - OCR 매칭(EasyOCR)

    단, 템플릿 경로/액션 좌표/크기 정보는 템플릿 매처를 기준으로 사용한다.
    (클릭 좌표 계산이 템플릿 데이터 기준이기 때문)
    """

    def __init__(self, mode: str = "ocr", threshold: float = 0.7):
        self.threshold = threshold
        self.mode = self._normalize_mode(mode)

        # 둘 다 미리 만들어두면 분기 코드가 단순해진다.
        self.template_matcher = TemplateImageMatcher(threshold=threshold)
        self.ocr_matcher = ImageMatcherEasyOCR(threshold=threshold)

    def _normalize_mode(self, mode: str) -> str:
        return mode if mode in {"ocr", "template"} else "ocr"

    def set_mode(self, mode: str) -> None:
        self.mode = self._normalize_mode(mode)

    @property
    def metadata_matcher(self):
        return self.template_matcher

    @property
    def active_matcher(self):
        if self.mode == "ocr":
            return self.ocr_matcher
        return self.template_matcher

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
        # 템플릿 파일은 template_matcher 쪽이 관리하므로 이것만 다시 만든다.
        self.template_matcher = TemplateImageMatcher(threshold=self.threshold)

    def _is_runtime_macro_template_id(self, template_id: str) -> bool:
        """
        런타임 매크로는 M1_A1 같은 ID를 사용한다.
        이런 경우 클릭 좌표 계산 때문에 템플릿 매처 결과를 강제로 사용한다.
        """
        return bool(re.match(r"^M\d+_A\d+$", str(template_id)))

    def find_template(self, template_id: str):
        if self._is_runtime_macro_template_id(template_id):
            return self.template_matcher.find_template(template_id)
        return self.active_matcher.find_template(template_id)

    def load_template(self, template_id: str):
        return self.metadata_matcher.load_template(template_id)

    def get_scaled_action_coordinates(
        self,
        template_id: str,
        action_id: str,
        template_location: Any,
        scale_info: Any,
    ):
        return self.metadata_matcher.get_scaled_action_coordinates(
            template_id,
            action_id,
            template_location,
            scale_info,
        )

    def get_action_center(
        self,
        template_id: str,
        action_id: str,
        template_location: Any,
        scale_info: Any,
    ):
        return self.metadata_matcher.get_action_center(
            template_id,
            action_id,
            template_location,
            scale_info,
        )
