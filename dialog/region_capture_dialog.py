from __future__ import annotations

from pathlib import Path

from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets


class CaptureCanvas(QtWidgets.QLabel):
    region_added = QtCore.pyqtSignal(dict)

    def __init__(self, pixmap: QtGui.QPixmap, single_select: bool = False, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self._base_pixmap = pixmap
        self._origin = None
        self._current_rect = None
        self._regions = []
        self._scale = 1.0
        self._single_select = single_select
        self._refresh_canvas()

    @property
    def regions(self):
        return self._regions

    @property
    def scale_factor(self):
        return self._scale

    def clear_regions(self):
        self._regions.clear()
        self.update()

    def _refresh_canvas(self):
        width = max(1, int(self._base_pixmap.width() * self._scale))
        height = max(1, int(self._base_pixmap.height() * self._scale))
        scaled = self._base_pixmap.scaled(
            width,
            height,
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.resize(scaled.size())
        self.update()

    def _to_image_point(self, point: QtCore.QPoint):
        x = int(point.x() / self._scale)
        y = int(point.y() / self._scale)
        x = max(0, min(x, self._base_pixmap.width() - 1))
        y = max(0, min(y, self._base_pixmap.height() - 1))
        return QtCore.QPoint(x, y)

    def set_scale(self, scale: float):
        scale = max(0.2, min(4.0, float(scale)))
        if abs(scale - self._scale) < 0.001:
            return
        self._scale = scale
        self._refresh_canvas()

    def zoom_in(self):
        self.set_scale(self._scale * 1.1)

    def zoom_out(self):
        self.set_scale(self._scale * 0.9)

    def reset_zoom(self):
        self.set_scale(1.0)

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        self._origin = self._to_image_point(event.position().toPoint())
        self._current_rect = QtCore.QRect(self._origin, self._origin)
        self.update()

    def mouseMoveEvent(self, event):
        if self._origin is None:
            return
        current = self._to_image_point(event.position().toPoint())
        self._current_rect = QtCore.QRect(self._origin, current).normalized()
        self.update()

    def mouseReleaseEvent(self, event):
        if self._origin is None or self._current_rect is None:
            return
        rect = self._current_rect.normalized()
        self._origin = None
        self._current_rect = None
        if rect.width() < 5 or rect.height() < 5:
            self.update()
            return

        region = {
            "capture_rect": {
                "x": rect.x(),
                "y": rect.y(),
                "width": rect.width(),
                "height": rect.height(),
            },
            "screen_rect": {
                "x": rect.x(),
                "y": rect.y(),
                "width": rect.width(),
                "height": rect.height(),
            },
            "item_type": "button",
        }
        if self._single_select:
            self._regions = [region]
        else:
            self._regions.append(region)
        self.region_added.emit(region)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtGui.QColor("red"), 2))
        for region in self._regions:
            rect = region["capture_rect"]
            painter.drawRect(
                int(rect["x"] * self._scale),
                int(rect["y"] * self._scale),
                int(rect["width"] * self._scale),
                int(rect["height"] * self._scale),
            )
        if self._current_rect:
            painter.drawRect(
                int(self._current_rect.x() * self._scale),
                int(self._current_rect.y() * self._scale),
                int(self._current_rect.width() * self._scale),
                int(self._current_rect.height() * self._scale),
            )


class RegionCaptureDialog(QtWidgets.QDialog):
    def __init__(self, screenshot_path: str, parent=None, single_select: bool = False):
        super().__init__(parent)
        self.setWindowTitle("영역 선택")
        self.resize(1100, 760)
        self.screenshot_path = screenshot_path
        self.capture_regions = []
        self.single_select = single_select

        root_layout = QtWidgets.QHBoxLayout(self)
        pixmap = QtGui.QPixmap(screenshot_path)
        self.canvas = CaptureCanvas(pixmap, single_select=single_select)
        self.canvas.region_added.connect(self._on_region_added)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)

        zoom_row = QtWidgets.QHBoxLayout()
        zoom_row.setSpacing(8)
        self.button_zoom_in = QtWidgets.QPushButton("확대")
        self.button_zoom_out = QtWidgets.QPushButton("축소")
        self.button_zoom_reset = QtWidgets.QPushButton("원본크기")
        self.label_zoom = QtWidgets.QLabel("100%")
        self.label_zoom.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_zoom.setMinimumWidth(56)
        zoom_row.addWidget(self.button_zoom_in)
        zoom_row.addWidget(self.button_zoom_out)
        zoom_row.addWidget(self.button_zoom_reset)
        zoom_row.addStretch(1)
        zoom_row.addWidget(self.label_zoom)
        left.addLayout(zoom_row)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.canvas)
        left.addWidget(scroll, 1)
        root_layout.addLayout(left, 1)

        side = QtWidgets.QVBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        side.addWidget(self.list_widget, 1)

        self.button_type = QtWidgets.QPushButton("선택 항목: 버튼/텍스트 전환")
        self.button_type.clicked.connect(self.toggle_selected_type)
        side.addWidget(self.button_type)

        self.button_delete = QtWidgets.QPushButton("선택 항목 삭제")
        self.button_delete.clicked.connect(self.delete_selected)
        side.addWidget(self.button_delete)

        self.button_save = QtWidgets.QPushButton("저장")
        self.button_save.clicked.connect(self.accept)
        side.addWidget(self.button_save)

        self.button_cancel = QtWidgets.QPushButton("취소")
        self.button_cancel.clicked.connect(self.reject)
        side.addWidget(self.button_cancel)
        root_layout.addLayout(side)

        self.button_zoom_in.clicked.connect(self._zoom_in)
        self.button_zoom_out.clicked.connect(self._zoom_out)
        self.button_zoom_reset.clicked.connect(self._reset_zoom)
        self._update_zoom_label()

        if self.single_select:
            self.button_type.hide()

    def _update_zoom_label(self):
        self.label_zoom.setText(f"{int(round(self.canvas.scale_factor * 100))}%")

    def _zoom_in(self):
        self.canvas.zoom_in()
        self._update_zoom_label()

    def _zoom_out(self):
        self.canvas.zoom_out()
        self._update_zoom_label()

    def _reset_zoom(self):
        self.canvas.reset_zoom()
        self._update_zoom_label()

    def _preview_path_for_region(self, index: int, rect: dict):
        preview_path = str(Path(self.screenshot_path).with_name(f"capture_region_{index}.png"))
        with Image.open(self.screenshot_path) as image:
            left = rect["x"]
            top = rect["y"]
            right = left + rect["width"]
            bottom = top + rect["height"]
            image.crop((left, top, right, bottom)).save(preview_path)
        return preview_path

    def _on_region_added(self, region: dict):
        if self.single_select:
            self.capture_regions = []

        index = len(self.capture_regions) + 1
        region["name"] = f"버튼 {index}"
        region["preview_image"] = self._preview_path_for_region(index, region["capture_rect"])
        self.capture_regions.append(region)
        self._refresh_list()

    def _refresh_list(self):
        self.list_widget.clear()
        for region in self.capture_regions:
            label = "텍스트" if region.get("item_type") == "text" else "버튼"
            self.list_widget.addItem(f"[{label}] {region.get('name')}")
        if self.capture_regions:
            self.list_widget.setCurrentRow(len(self.capture_regions) - 1)

    def toggle_selected_type(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.capture_regions):
            return
        region = self.capture_regions[row]
        if region.get("item_type") == "text":
            region["item_type"] = "button"
            region["name"] = f"버튼 {row + 1}"
        else:
            region["item_type"] = "text"
            region["name"] = f"텍스트 {row + 1}"
        self._refresh_list()

    def delete_selected(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.capture_regions):
            return
        self.capture_regions.pop(row)
        self.canvas.regions.pop(row)
        self.canvas.update()
        self._refresh_list()
