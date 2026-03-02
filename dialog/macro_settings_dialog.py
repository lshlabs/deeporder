from __future__ import annotations

from PyQt6 import QtWidgets


class MacroSettingsDialog(QtWidgets.QDialog):
    def __init__(self, macro_settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("매크로 설정")
        self.setModal(True)
        self.resize(320, 180)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        repeat_row = QtWidgets.QHBoxLayout()
        repeat_label = QtWidgets.QLabel("실행 횟수")
        repeat_label.setFixedWidth(80)
        self.spin_repeat = QtWidgets.QSpinBox(self)
        self.spin_repeat.setRange(0, 9999)
        self.spin_repeat.setValue(int(macro_settings.get("repeat_count", 1) or 0))
        repeat_hint = QtWidgets.QLabel("0 = 무한 반복")
        repeat_hint.setStyleSheet("color:#4b5563;")
        repeat_row.addWidget(repeat_label)
        repeat_row.addWidget(self.spin_repeat)
        repeat_row.addWidget(repeat_hint, 1)
        root_layout.addLayout(repeat_row)

        delay_row = QtWidgets.QHBoxLayout()
        delay_label = QtWidgets.QLabel("반복 딜레이")
        delay_label.setFixedWidth(80)
        self.spin_delay = QtWidgets.QDoubleSpinBox(self)
        self.spin_delay.setRange(0.0, 9999.0)
        self.spin_delay.setDecimals(1)
        self.spin_delay.setSingleStep(0.1)
        self.spin_delay.setValue(float(macro_settings.get("repeat_delay_sec", 0.5) or 0.0))
        delay_suffix = QtWidgets.QLabel("초")
        delay_row.addWidget(delay_label)
        delay_row.addWidget(self.spin_delay)
        delay_row.addWidget(delay_suffix)
        delay_row.addStretch(1)
        root_layout.addLayout(delay_row)

        footer_row = QtWidgets.QHBoxLayout()
        footer_row.addStretch(1)
        self.button_save = QtWidgets.QPushButton("저장", self)
        self.button_cancel = QtWidgets.QPushButton("취소", self)
        footer_row.addWidget(self.button_save)
        footer_row.addWidget(self.button_cancel)
        root_layout.addLayout(footer_row)

        self.button_save.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

    def values(self) -> tuple[int, float]:
        return self.spin_repeat.value(), self.spin_delay.value()
