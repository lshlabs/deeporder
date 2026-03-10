from __future__ import annotations

from PyQt6 import QtWidgets, uic

from utils.path_manager import ui_path


class MacroSettingsDialog(QtWidgets.QDialog):
    def __init__(self, macro_settings: dict, parent=None):
        super().__init__(parent)
        uic.loadUi(str(ui_path("MacroSettingsDialog.ui")), self)
        self.spin_repeat = self.findChild(QtWidgets.QSpinBox, "spin_repeat")
        self.spin_delay = self.findChild(QtWidgets.QDoubleSpinBox, "spin_delay")
        self.button_save = self.findChild(QtWidgets.QPushButton, "button_save")
        self.button_cancel = self.findChild(QtWidgets.QPushButton, "button_cancel")

        self.spin_repeat.setValue(int(macro_settings.get("repeat_count", 1) or 0))
        self.spin_delay.setValue(float(macro_settings.get("repeat_delay_sec", 0.5) or 0.0))
        self.button_save.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

    def values(self) -> tuple[int, float]:
        return self.spin_repeat.value(), self.spin_delay.value()
