from __future__ import annotations

from PyQt6 import QtCore, QtWidgets, uic

from utils.path_manager import ui_path


class PresetTabLabel(QtWidgets.QPushButton):
    double_clicked = QtCore.pyqtSignal()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


class PresetRibbon(QtWidgets.QFrame):
    tab_selected = QtCore.pyqtSignal(str)
    tab_close_requested = QtCore.pyqtSignal(str)
    tab_rename_requested = QtCore.pyqtSignal(str)
    add_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(str(ui_path("PresetRibbon.ui")), self)
        self._chip_height = 28
        self.setFixedHeight(self._chip_height)
        self._layout = self.layout()

    def _clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_tabs(self, tabs: list[dict[str, str]], current_preset_id: str | None, default_preset_id: str | None):
        self._clear()
        for tab in tabs:
            preset_id = tab["id"]
            chip = QtWidgets.QFrame(self)
            chip.setObjectName("preset_chip")
            chip.setProperty("selected", preset_id == current_preset_id)
            chip.setFixedHeight(self._chip_height)

            chip_layout = QtWidgets.QHBoxLayout(chip)
            chip_layout.setContentsMargins(0, 0, 5, 0)
            chip_layout.setSpacing(3)
            chip_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

            label = PresetTabLabel(tab["name"], chip)
            label.setObjectName("preset_label")
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
            label.setMinimumHeight(22)
            label.setMaximumHeight(22)
            label.clicked.connect(lambda _checked=False, pid=preset_id: self.tab_selected.emit(pid))
            label.double_clicked.connect(lambda pid=preset_id: self.tab_rename_requested.emit(pid))
            chip_layout.addWidget(label)

            if preset_id != default_preset_id:
                close_button = QtWidgets.QPushButton("x", chip)
                close_button.setObjectName("preset_close")
                close_button.clicked.connect(lambda _checked=False, pid=preset_id: self.tab_close_requested.emit(pid))
                chip_layout.addWidget(close_button)

            chip.style().unpolish(chip)
            chip.style().polish(chip)
            self._layout.addWidget(chip, 0)

        add_button = QtWidgets.QPushButton("+", self)
        add_button.setObjectName("preset_add")
        add_button.setFixedHeight(self._chip_height)
        add_button.clicked.connect(self.add_requested.emit)
        self._layout.addWidget(add_button, 0)
        self._layout.addStretch(1)
