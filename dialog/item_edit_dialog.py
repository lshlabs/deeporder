from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets, uic

from utils.path_manager import ui_path


class ItemEditDialog(QtWidgets.QDialog):
    def __init__(self, item: dict, click_count: int | None = None, parent=None):
        super().__init__(parent)
        self.item = item
        self._build_ui(click_count)

    def _build_ui(self, click_count: int | None):
        uic.loadUi(str(ui_path("ItemEditDialog.ui")), self)
        self.setWindowTitle("항목 수정")
        self.setModal(True)
        target_height = 500 if self.item.get("item_type") != "text" else 450
        self.resize(max(self.width(), 460), target_height)

        self.preview_label = self.findChild(QtWidgets.QLabel, "preview_label")
        self.line_name = self.findChild(QtWidgets.QLineEdit, "line_name")
        self.spin_x = self.findChild(QtWidgets.QSpinBox, "spin_x")
        self.spin_y = self.findChild(QtWidgets.QSpinBox, "spin_y")
        self.spin_w = self.findChild(QtWidgets.QSpinBox, "spin_w")
        self.spin_h = self.findChild(QtWidgets.QSpinBox, "spin_h")
        self.spin_clicks = self.findChild(QtWidgets.QSpinBox, "spin_clicks")
        self.clicks_label = self.findChild(QtWidgets.QLabel, "clicks_label")
        self.button_save = self.findChild(QtWidgets.QPushButton, "button_save")
        self.button_cancel = self.findChild(QtWidgets.QPushButton, "button_cancel")

        preview_path = self.item.get("preview_image")
        if preview_path:
            pixmap = QtGui.QPixmap(preview_path)
            if not pixmap.isNull():
                preview_size = self.preview_label.size()
                scaled = pixmap.scaled(
                    max(160, preview_size.width() - 12),
                    max(96, preview_size.height() - 12),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(scaled)

        rect = self.item.get("screen_rect") or self.item.get("capture_rect")
        for spin_box in (self.spin_x, self.spin_y, self.spin_w, self.spin_h):
            spin_box.setRange(0, 99999)
            spin_box.setMinimumHeight(34)

        if rect:
            self.spin_x.setValue(int(rect.get("x", 0) or 0))
            self.spin_y.setValue(int(rect.get("y", 0) or 0))
            self.spin_w.setValue(int(rect.get("width", 0) or 0))
            self.spin_h.setValue(int(rect.get("height", 0) or 0))

        self.line_name.setText(self.item.get("name", ""))
        if self.item.get("item_type") != "text":
            self.spin_clicks.setRange(1, 99)
            self.spin_clicks.setValue(max(1, int(click_count or 1)))
        else:
            self.clicks_label.hide()
            self.spin_clicks.hide()
            self.resize(max(self.width(), 460), 450)

        self.button_save.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

    def get_values(self) -> tuple[str, int | None, dict[str, int]]:
        name = self.line_name.text().strip() or self.item.get("name", "")
        clicks = self.spin_clicks.value() if self.spin_clicks.isVisible() else None
        rect = {
            "x": self.spin_x.value(),
            "y": self.spin_y.value(),
            "width": self.spin_w.value(),
            "height": self.spin_h.value(),
        }
        return name, clicks, rect
