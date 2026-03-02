from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageGrab
from PyQt6 import QtCore, QtGui, QtWidgets

from dialog.region_capture_dialog import RegionCaptureDialog


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
        self.setObjectName("TriggerCard")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(10)
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("QLabel { font: 700 10pt 'Malgun Gothic'; color: #111827; }")
        header_row.addWidget(title_label, 1)
        self.check_enabled = QtWidgets.QCheckBox("활성화")
        header_row.addWidget(self.check_enabled, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addLayout(header_row)

        type_row = QtWidgets.QHBoxLayout()
        type_row.setSpacing(10)
        type_label = QtWidgets.QLabel("타입")
        type_label.setFixedWidth(46)
        self.combo_type = NoWheelComboBox()
        self.combo_type.addItem("사용 안 함", None)
        self.combo_type.addItem("픽셀 색상", "pixel_color")
        self.combo_type.addItem("텍스트 감지", "text_roi")
        type_row.addWidget(type_label)
        type_row.addWidget(self.combo_type, 1)
        layout.addLayout(type_row)

        self.text_item_row = QtWidgets.QHBoxLayout()
        self.text_item_row.setSpacing(10)
        self.text_item_label = QtWidgets.QLabel("요소")
        self.text_item_label.setFixedWidth(46)
        self.combo_text_item = NoWheelComboBox()
        self.combo_text_item.addItem("직접 선택")
        for item in self.text_items:
            self.combo_text_item.addItem(item.get("name", item.get("id", "텍스트 요소")), item)
        self.text_item_row.addWidget(self.text_item_label)
        self.text_item_row.addWidget(self.combo_text_item, 1)
        layout.addLayout(self.text_item_row)

        self.guide_label = QtWidgets.QLabel("")
        self.guide_label.setWordWrap(True)
        self.guide_label.setStyleSheet("QLabel { color: #4b5563; font: 8.5pt 'Malgun Gothic'; padding-left: 2px; }")
        layout.addWidget(self.guide_label)

        self.region_row = QtWidgets.QHBoxLayout()
        self.region_row.setSpacing(10)
        self.button_region = QtWidgets.QPushButton("영역 지정")
        self.region_label = QtWidgets.QLabel("선택된 영역 없음")
        self.region_label.setWordWrap(True)
        self.region_label.setMinimumHeight(36)
        self.region_label.setStyleSheet(
            "QLabel { color: #4b5563; background: white; border: 1px solid #d1d5db; border-radius: 10px; padding: 0 10px; }"
        )
        self.region_row.addWidget(self.button_region)
        self.region_row.addWidget(self.region_label, 1)
        layout.addLayout(self.region_row)

        self.color_row = QtWidgets.QHBoxLayout()
        self.color_row.setSpacing(10)
        self.button_color = QtWidgets.QPushButton("색상 고르기")
        self.color_chip = QtWidgets.QLabel("미선택")
        self.color_chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.color_chip.setFixedSize(92, 32)
        self.color_chip.setStyleSheet(
            "QLabel { background: #f3f4f6; color: #6b7280; border: 1px solid #d1d5db; border-radius: 8px; }"
        )
        self.button_sample = QtWidgets.QPushButton("중앙색 추출")
        self.color_row.addWidget(self.button_color)
        self.color_row.addWidget(self.color_chip)
        self.color_row.addWidget(self.button_sample)
        layout.addLayout(self.color_row)

        self.text_row = QtWidgets.QHBoxLayout()
        self.text_row.setSpacing(10)
        text_label = QtWidgets.QLabel("텍스트")
        text_label.setFixedWidth(46)
        self.line_text = QtWidgets.QLineEdit()
        self.line_text.setPlaceholderText("예: 15분")
        self.text_row.addWidget(text_label)
        self.text_row.addWidget(self.line_text, 1)
        layout.addLayout(self.text_row)

        self.combo_type.currentTextChanged.connect(self._sync_visible_fields)
        self.combo_text_item.currentIndexChanged.connect(self.apply_text_item_selection)
        self.button_region.clicked.connect(self.select_region)
        self.button_color.clicked.connect(self.pick_color)
        self.button_sample.clicked.connect(self.sample_color_from_region)

        self.load_trigger(trigger)
        self._sync_visible_fields()

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
        self._set_row_visible(self.region_row, active and (not text_enabled or not has_text_items or not using_existing_text_item))
        self._set_row_visible(self.color_row, color_enabled)
        self._set_row_visible(self.text_row, text_enabled and (not has_text_items or not using_existing_text_item))
        self._set_row_visible(self.text_item_row, text_enabled and has_text_items)
        self._sync_guide_label(trigger_type)

        if text_enabled and self.text_items and self.combo_text_item.currentIndex() == 0 and not self.trigger_rect:
            self.combo_text_item.setCurrentIndex(1)

    def _set_row_visible(self, row_layout: QtWidgets.QLayout, visible: bool):
        for index in range(row_layout.count()):
            item = row_layout.itemAt(index)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(visible)

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
        except Exception as e:
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
        except Exception as e:
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
        self.setStyleSheet(
            "QDialog { background: #f3f4f6; }"
            "QFrame#DialogCard { background: white; border: 1px solid #d1d5db; border-radius: 14px; }"
            "QFrame#TriggerCard { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; }"
            "QPushButton {"
            "  min-height: 38px;"
            "  padding: 0 14px;"
            "  border-radius: 10px;"
            "  border: 1px solid #d1d5db;"
            "  background: white;"
            "  color: #111827;"
            "  font: 9pt 'Malgun Gothic';"
            "}"
            "QPushButton#PrimaryButton { background: #0ea5e9; border-color: #0284c7; color: white; font-weight: 700; }"
            "QLabel, QCheckBox, QComboBox, QLineEdit { color: #111827; font: 9pt 'Malgun Gothic'; }"
            "QComboBox, QLineEdit {"
            "  min-height: 36px;"
            "  border-radius: 10px;"
            "  border: 1px solid #d1d5db;"
            "  background: white;"
            "  padding-left: 10px;"
            "}"
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        card = QtWidgets.QFrame(self)
        card.setObjectName("DialogCard")
        root.addWidget(card)

        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(12)

        title = QtWidgets.QLabel("트리거 설정")
        title.setStyleSheet("QLabel { font: 700 12pt 'Malgun Gothic'; }")
        card_layout.addWidget(title)

        if self.macro.get("capture_source_image"):
            subtitle_text = "매크로 생성 때 저장한 화면을 기준으로 영역과 색상을 선택해 트리거를 설정합니다."
        else:
            subtitle_text = "저장된 캡처 화면이 없으면 현재 화면을 기준으로 영역과 색상을 선택합니다."
        subtitle = QtWidgets.QLabel(subtitle_text)
        subtitle.setStyleSheet("QLabel { color: #4b5563; font: 9pt 'Malgun Gothic'; }")
        card_layout.addWidget(subtitle)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        card_layout.addWidget(scroll, 1)

        scroll_host = QtWidgets.QWidget()
        scroll.setWidget(scroll_host)
        scroll_layout = QtWidgets.QVBoxLayout(scroll_host)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        shared_screenshot = self.macro.get("capture_source_image")

        text_items = [
            item
            for item in self.macro.get("items", {}).values()
            if isinstance(item, dict) and item.get("item_type") == "text"
        ]

        self.macro_editor = TriggerEditorWidget(
            "매크로 실행 트리거",
            self.macro.get("macro_trigger"),
            screenshot_path=shared_screenshot,
        )
        scroll_layout.addWidget(self.macro_editor)

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
            scroll_layout.addWidget(editor)

        scroll_layout.addStretch(1)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.button_save = QtWidgets.QPushButton("저장")
        self.button_save.setObjectName("PrimaryButton")
        self.button_cancel = QtWidgets.QPushButton("취소")
        footer.addWidget(self.button_save)
        footer.addWidget(self.button_cancel)
        card_layout.addLayout(footer)

        self.button_save.clicked.connect(self._save)
        self.button_cancel.clicked.connect(self.reject)

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
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
