from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageGrab
from PyQt6 import QtCore, QtGui, QtWidgets, uic

from dialog.region_capture_dialog import RegionCaptureDialog
from utils.path_manager import ui_path


class NoWheelComboBox(QtWidgets.QComboBox):
    def wheelEvent(self, event: QtGui.QWheelEvent):
        event.ignore()


class TriggerEditorWidget(QtWidgets.QFrame):
    def __init__(
        self,
        title: str,
        trigger: dict | None = None,
        required: bool = False,
        screenshot_path: str | None = None,
        text_items: list[dict] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.required = required
        self.screenshot_path = screenshot_path
        self.text_items = text_items or []
        self.trigger_rect = None
        self.trigger_color = None
        uic.loadUi(str(ui_path("TriggerEditorWidget.ui")), self)
        self.setObjectName("TriggerCard")

        self.label_title = self.findChild(QtWidgets.QLabel, "label_title")
        self.check_enabled = self.findChild(QtWidgets.QCheckBox, "check_enabled")
        self.combo_type = self._replace_combo_with_no_wheel("combo_type")
        self.combo_text_item = self._replace_combo_with_no_wheel("combo_text_item")
        self.guide_label = self.findChild(QtWidgets.QLabel, "guide_label")
        self.button_region = self.findChild(QtWidgets.QPushButton, "button_region")
        self.region_label = self.findChild(QtWidgets.QLabel, "region_label")
        self.button_color = self.findChild(QtWidgets.QPushButton, "button_color")
        self.color_chip = self.findChild(QtWidgets.QLabel, "color_chip")
        self.button_sample = self.findChild(QtWidgets.QPushButton, "button_sample")
        self.line_text = self.findChild(QtWidgets.QLineEdit, "line_text")

        self.row_text_item = self.findChild(QtWidgets.QWidget, "row_text_item")
        self.row_region = self.findChild(QtWidgets.QWidget, "row_region")
        self.row_color = self.findChild(QtWidgets.QWidget, "row_color")
        self.row_text = self.findChild(QtWidgets.QWidget, "row_text")

        self.label_title.setText(title)
        self.combo_type.clear()
        self.combo_type.addItem("사용 안 함", None)
        self.combo_type.addItem("픽셀 색상", "pixel_color")
        self.combo_type.addItem("텍스트 감지", "text_roi")

        self.combo_text_item.clear()
        self.combo_text_item.addItem("직접 선택")
        for item in self.text_items:
            self.combo_text_item.addItem(item.get("name", item.get("id", "텍스트 요소")), item)

        self.combo_type.currentTextChanged.connect(self._sync_visible_fields)
        self.combo_text_item.currentIndexChanged.connect(self.apply_text_item_selection)
        self.button_region.clicked.connect(self.select_region)
        self.button_color.clicked.connect(self.pick_color)
        self.button_sample.clicked.connect(self.sample_color_from_region)

        self.load_trigger(trigger)
        self._sync_visible_fields()

    def _replace_combo_with_no_wheel(self, object_name: str) -> NoWheelComboBox:
        original_combo = self.findChild(QtWidgets.QComboBox, object_name)
        if original_combo is None:
            raise RuntimeError(f"{object_name} 콤보박스를 찾을 수 없습니다.")

        parent_widget = original_combo.parentWidget()
        layout = parent_widget.layout() if parent_widget is not None else None
        if layout is None:
            raise RuntimeError(f"{object_name}의 부모 레이아웃을 찾을 수 없습니다.")

        index = layout.indexOf(original_combo)
        replacement = NoWheelComboBox(parent_widget)
        replacement.setObjectName(object_name)
        replacement.setMinimumSize(original_combo.minimumSize())
        replacement.setMaximumSize(original_combo.maximumSize())
        replacement.setSizePolicy(original_combo.sizePolicy())
        replacement.setEnabled(original_combo.isEnabled())

        layout.removeWidget(original_combo)
        original_combo.deleteLater()
        layout.insertWidget(index, replacement, 1)
        return replacement

    def load_trigger(self, trigger: dict | None):
        if not trigger:
            self.combo_type.setCurrentIndex(0)
            self.check_enabled.setChecked(False)
            self._set_region(None)
            self._set_color(None)
            self.line_text.clear()
            return

        trigger_type = trigger.get("trigger_type")
        index = 0
        for current_index in range(self.combo_type.count()):
            if self.combo_type.itemData(current_index) == trigger_type:
                index = current_index
                break
        self.combo_type.setCurrentIndex(index)
        self.check_enabled.setChecked(bool(trigger.get("enabled", False)))
        self.line_text.setText(str(trigger.get("expected_text") or ""))
        self._set_region(trigger.get("screen_rect"))
        self._set_color(trigger.get("expected_color"))

    def _sync_visible_fields(self):
        trigger_type = self.combo_type.currentData()
        color_enabled = trigger_type == "pixel_color"
        text_enabled = trigger_type == "text_roi"
        active = bool(trigger_type)
        has_text_items = bool(self.text_items)
        using_existing_text_item = text_enabled and has_text_items and self.combo_text_item.currentIndex() > 0

        self.button_region.setEnabled(active)
        self.region_label.setEnabled(active)
        self._set_row_visible(self.row_region, active and (not text_enabled or not has_text_items or not using_existing_text_item))
        self._set_row_visible(self.row_color, color_enabled)
        self._set_row_visible(self.row_text, text_enabled and (not has_text_items or not using_existing_text_item))
        self._set_row_visible(self.row_text_item, text_enabled and has_text_items)
        self._sync_guide_label(trigger_type)

        if text_enabled and self.text_items and self.combo_text_item.currentIndex() == 0 and not self.trigger_rect:
            self.combo_text_item.setCurrentIndex(1)

    @staticmethod
    def _set_row_visible(row_widget: QtWidgets.QWidget, visible: bool):
        row_widget.setVisible(visible)

    def _sync_guide_label(self, trigger_type):
        if trigger_type == "text_roi":
            if self.text_items and self.combo_text_item.currentIndex() > 0:
                self.guide_label.setText("선택한 텍스트 요소의 영역과 이름을 그대로 사용합니다. 필요하면 '직접 선택'으로 바꿔 수동 입력하세요.")
            elif self.text_items:
                self.guide_label.setText("기존 텍스트 요소를 먼저 고르거나, 직접 영역을 지정한 뒤 감지할 문구를 입력하세요.")
            else:
                self.guide_label.setText("텍스트 요소가 없습니다. 영역을 지정한 뒤 감지할 문구를 입력하세요.")
            self.guide_label.show()
            return
        if trigger_type == "pixel_color":
            self.guide_label.setText("영역을 지정한 뒤 색상을 고르거나 중앙색을 바로 가져오세요.")
            self.guide_label.show()
            return
        self.guide_label.hide()

    def _set_region(self, rect):
        self.trigger_rect = rect
        if not rect:
            self.region_label.setText("선택된 영역 없음")
            return
        self.region_label.setText(f"({rect['x']}, {rect['y']}) {rect['width']} x {rect['height']}")

    def _set_color(self, color):
        self.trigger_color = color
        if not color:
            self.color_chip.setText("미선택")
            self.color_chip.setStyleSheet(
                "QLabel { background: #f3f4f6; color: #6b7280; border: 1px solid #d1d5db; border-radius: 8px; }"
            )
            return
        qcolor = QtGui.QColor(int(color["r"]), int(color["g"]), int(color["b"]))
        text_color = "#111827" if qcolor.lightness() > 140 else "white"
        self.color_chip.setText(f"{color['r']}, {color['g']}, {color['b']}")
        self.color_chip.setStyleSheet(
            f"QLabel {{ background: {qcolor.name()}; color: {text_color}; border: 1px solid #d1d5db; border-radius: 8px; }}"
        )

    def _capture_fullscreen(self):
        screenshot = ImageGrab.grab(all_screens=True)
        temp_dir = Path(__file__).resolve().parents[1] / "temp"
        temp_dir.mkdir(exist_ok=True)
        path = temp_dir / "trigger_capture.png"
        screenshot.save(path)
        return path

    def _get_capture_source_path(self):
        if self.screenshot_path:
            existing = Path(self.screenshot_path)
            if existing.exists():
                return existing
        return self._capture_fullscreen()

    def select_region(self):
        try:
            screenshot_path = self._get_capture_source_path()
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "캡처 오류", f"화면 캡처에 실패했습니다.\n{e}")
            return

        dialog = RegionCaptureDialog(str(screenshot_path), self, single_select=True)
        dialog.setWindowTitle("트리거 영역 선택")
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dialog.capture_regions:
            return
        first = dialog.capture_regions[0]
        self._set_region(first.get("screen_rect"))
        if self.combo_type.currentData() == "pixel_color":
            self._sample_center_color_from_path(screenshot_path, self.trigger_rect)

    def apply_text_item_selection(self):
        if self.combo_type.currentData() != "text_roi":
            return
        item = self.combo_text_item.currentData()
        if not item:
            self._sync_visible_fields()
            return
        rect = item.get("screen_rect") or item.get("capture_rect")
        if rect:
            self._set_region(
                {
                    "x": int(rect.get("x", 0) or 0),
                    "y": int(rect.get("y", 0) or 0),
                    "width": int(rect.get("width", 0) or 0),
                    "height": int(rect.get("height", 0) or 0),
                }
            )
        self.line_text.setText(str(item.get("name") or ""))
        self._sync_visible_fields()

    def pick_color(self):
        initial = QtGui.QColor(255, 255, 255)
        if self.trigger_color:
            initial = QtGui.QColor(
                int(self.trigger_color["r"]),
                int(self.trigger_color["g"]),
                int(self.trigger_color["b"]),
            )
        selected = QtWidgets.QColorDialog.getColor(initial, self, "색상 선택")
        if not selected.isValid():
            return
        self._set_color({"r": selected.red(), "g": selected.green(), "b": selected.blue()})

    def _sample_center_color_from_path(self, screenshot_path: Path, rect: dict | None):
        if not rect:
            return
        with Image.open(screenshot_path) as image:
            cx = int(rect["x"]) + int(rect["width"]) // 2
            cy = int(rect["y"]) + int(rect["height"]) // 2
            cx = max(0, min(cx, image.width - 1))
            cy = max(0, min(cy, image.height - 1))
            pixel = image.convert("RGB").getpixel((cx, cy))
        self._set_color({"r": int(pixel[0]), "g": int(pixel[1]), "b": int(pixel[2])})

    def sample_color_from_region(self):
        if not self.trigger_rect:
            QtWidgets.QMessageBox.information(self, "안내", "먼저 영역을 선택해주세요.")
            return
        try:
            screenshot_path = self._get_capture_source_path()
            self._sample_center_color_from_path(screenshot_path, self.trigger_rect)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "캡처 오류", f"색상 추출에 실패했습니다.\n{e}")

    def to_trigger(self):
        trigger_type = self.combo_type.currentData()
        if not trigger_type:
            return None

        if not self.trigger_rect:
            raise ValueError("영역을 선택해주세요.")

        trigger = {
            "trigger_type": trigger_type,
            "screen_rect": self.trigger_rect,
            "expected_color": self.trigger_color,
            "color_tolerance": 10,
            "expected_text": self.line_text.text().strip() or None,
            "text_match_mode": "contains",
            "ocr_confidence_min": 0.6,
            "enabled": self.check_enabled.isChecked(),
        }
        if trigger_type == "text_roi" and self.text_items and self.combo_text_item.currentIndex() > 0:
            selected_item = self.combo_text_item.currentData()
            selected_name = str((selected_item or {}).get("name") or "").strip()
            if selected_name:
                trigger["expected_text"] = selected_name
        if trigger_type == "pixel_color" and not trigger["expected_color"]:
            raise ValueError("색상을 선택해주세요.")
        if trigger_type == "text_roi" and not trigger["expected_text"]:
            raise ValueError("텍스트를 입력해주세요.")
        return trigger


class TriggerDialog(QtWidgets.QDialog):
    def __init__(self, macro: dict, current_preset_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("트리거 설정")
        self.resize(760, 760)
        self.macro = macro
        self.current_preset_id = current_preset_id
        self.preset_editors = {}
        self.result_payload = None
        self._build_ui()

    def _build_ui(self):
        uic.loadUi(str(ui_path("TriggerDialog.ui")), self)
        self.label_subtitle = self.findChild(QtWidgets.QLabel, "label_subtitle")
        self.scroll_area = self.findChild(QtWidgets.QScrollArea, "scroll_area")
        self.scroll_host = self.findChild(QtWidgets.QWidget, "scroll_host")
        self.button_save = self.findChild(QtWidgets.QPushButton, "button_save")
        self.button_cancel = self.findChild(QtWidgets.QPushButton, "button_cancel")

        self.scroll_layout = self.scroll_host.layout()
        subtitle_text = (
            "매크로 생성 때 저장한 화면을 기준으로 영역과 색상을 선택해 트리거를 설정합니다."
            if self.macro.get("capture_source_image")
            else "저장된 캡처 화면이 없으면 현재 화면을 기준으로 영역과 색상을 선택합니다."
        )
        self.label_subtitle.setText(subtitle_text)
        self._build_editors()
        self.button_save.clicked.connect(self._save)
        self.button_cancel.clicked.connect(self.reject)

    def _build_editors(self) -> None:
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        shared_screenshot = self.macro.get("capture_source_image")
        text_items = self._collect_text_items()

        self.macro_editor = TriggerEditorWidget(
            "매크로 실행 트리거",
            self.macro.get("macro_trigger"),
            screenshot_path=shared_screenshot,
        )
        self.scroll_layout.addWidget(self.macro_editor)

        for preset_id, preset in self.macro.get("presets", {}).items():
            required = preset_id != self.macro.get("default_preset_id")
            editor = TriggerEditorWidget(
                f"{preset.get('name', preset_id)} 프리셋 실행 트리거",
                preset.get("preset_trigger"),
                required=required,
                screenshot_path=shared_screenshot,
                text_items=text_items,
            )
            self.preset_editors[preset_id] = editor
            self.scroll_layout.addWidget(editor)

        self.scroll_layout.addStretch(1)

    def _collect_text_items(self) -> list[dict]:
        return [
            item
            for item in self.macro.get("items", {}).values()
            if isinstance(item, dict) and item.get("item_type") == "text"
        ]

    def _save(self):
        try:
            payload = {"macro_trigger": self.macro_editor.to_trigger(), "preset_triggers": {}}
            default_id = self.macro.get("default_preset_id")
            for preset_id, editor in self.preset_editors.items():
                trigger = editor.to_trigger()
                if preset_id != default_id and trigger is None:
                    raise ValueError(f"{self.macro['presets'][preset_id].get('name', preset_id)} 트리거가 필요합니다.")
                payload["preset_triggers"][preset_id] = trigger
            self.result_payload = payload
            self.accept()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
