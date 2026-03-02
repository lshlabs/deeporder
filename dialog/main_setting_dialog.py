from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets, uic

from utils.data_manager import DataManager
from utils.path_manager import ui_path


class HotkeyLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setPlaceholderText("키를 누르세요")

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()
        if key in {QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete}:
            self.clear()
            event.accept()
            return

        if key in {
            QtCore.Qt.Key.Key_Control,
            QtCore.Qt.Key.Key_Shift,
            QtCore.Qt.Key.Key_Alt,
            QtCore.Qt.Key.Key_Meta,
            QtCore.Qt.Key.Key_unknown,
        }:
            event.accept()
            return

        parts: list[str] = []
        modifiers = event.modifiers()
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            parts.append("Ctrl")
        if modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
            parts.append("Alt")
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            parts.append("Shift")
        if modifiers & QtCore.Qt.KeyboardModifier.MetaModifier:
            parts.append("Meta")

        key_text = QtGui.QKeySequence(key).toString().strip()
        if not key_text:
            key_text = event.text().upper().strip()
        if key_text:
            if len(key_text) == 1 or key_text.upper().startswith("F"):
                key_text = key_text.upper()
            parts.append(key_text)
            self.setText("+".join(parts))

        event.accept()


class MainSettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, force_setup: bool = False):
        super().__init__(parent)
        uic.loadUi(str(ui_path("MainsettingWindow.ui")), self)
        self.data_manager = DataManager.get_instance()
        self.force_setup = force_setup
        self._build_ui()
        self._connect_signals()
        self.load_settings()

    def _build_ui(self):
        self.setFixedSize(380, 400)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)

        self.frame_main = self.findChild(QtWidgets.QFrame, "frame_main")
        self.comboBox_resolution = self.findChild(QtWidgets.QComboBox, "comboBox_run")
        self.lineEdit_width = self.findChild(QtWidgets.QLineEdit, "lineEdit_width")
        self.lineEdit_height = self.findChild(QtWidgets.QLineEdit, "lineEdit_height")
        self.label_width = self.findChild(QtWidgets.QLabel, "label_width")
        self.label_height = self.findChild(QtWidgets.QLabel, "label_height")
        self.label_tip = self.findChild(QtWidgets.QLabel, "label_tip")
        self.button_save = self.findChild(QtWidgets.QPushButton, "button_save")
        self.button_cancel = self.findChild(QtWidgets.QPushButton, "button_cancel")

        for widget in (self.lineEdit_width, self.lineEdit_height, self.label_width, self.label_height):
            widget.hide()

        self.lineEdit_hotkey = HotkeyLineEdit(self.frame_main)
        self.lineEdit_run_hotkey = HotkeyLineEdit(self.frame_main)
        self.lineEdit_stop_hotkey = HotkeyLineEdit(self.frame_main)
        self.label_status = QtWidgets.QLabel(self.frame_main)
        self.label_status.setWordWrap(True)

        self._apply_styles()
        self._build_layout()

        if self.force_setup:
            self.button_cancel.setEnabled(False)

    def _apply_styles(self):
        for hotkey_edit in (self.lineEdit_hotkey, self.lineEdit_run_hotkey, self.lineEdit_stop_hotkey):
            hotkey_edit.setMinimumHeight(30)
            hotkey_edit.setStyleSheet(
                "color:black; border-radius: 5px; padding-left:3px; background:white;"
            )
        self.label_status.setStyleSheet("border:none; color:black;")

    def _build_layout(self):
        self.frame_main.setGeometry(10, 10, 360, 380)

        old_layout = self.frame_main.layout()
        if old_layout is not None:
            while old_layout.count():
                old_layout.takeAt(0)

        layout = QtWidgets.QVBoxLayout(self.frame_main)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(14)

        self.label_tip.setParent(self.frame_main)
        layout.addWidget(self.label_tip)

        self.comboBox_resolution.setParent(self.frame_main)
        self.comboBox_resolution.setMinimumHeight(28)
        layout.addWidget(self.comboBox_resolution)

        self.custom_resolution_row = QtWidgets.QWidget(self.frame_main)
        custom_layout = QtWidgets.QHBoxLayout(self.custom_resolution_row)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.setSpacing(8)
        self.label_width.setParent(self.custom_resolution_row)
        self.lineEdit_width.setParent(self.custom_resolution_row)
        self.label_height.setParent(self.custom_resolution_row)
        self.lineEdit_height.setParent(self.custom_resolution_row)
        custom_layout.addWidget(self.label_width)
        custom_layout.addWidget(self.lineEdit_width, 1)
        custom_layout.addWidget(self.label_height)
        custom_layout.addWidget(self.lineEdit_height, 1)
        layout.addWidget(self.custom_resolution_row)
        self.custom_resolution_row.hide()

        layout.addLayout(self._build_labeled_row("캡처 핫키", self.lineEdit_hotkey))
        layout.addLayout(self._build_labeled_row("실행 핫키", self.lineEdit_run_hotkey))
        layout.addLayout(self._build_labeled_row("종료 핫키", self.lineEdit_stop_hotkey))

        layout.addWidget(self.label_status)
        layout.addStretch(1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self.button_save.setParent(self.frame_main)
        self.button_cancel.setParent(self.frame_main)
        button_row.addWidget(self.button_save)
        button_row.addWidget(self.button_cancel)
        layout.addLayout(button_row)

    def _build_labeled_row(self, label_text: str, editor: QtWidgets.QWidget) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        label = QtWidgets.QLabel(label_text, self.frame_main)
        label.setFixedWidth(70)
        label.setStyleSheet("border:none; color:black;")
        row.addWidget(label)
        row.addWidget(editor, 1)
        return row

    def _connect_signals(self):
        self.button_save.clicked.connect(self.save_settings)
        self.button_cancel.clicked.connect(self.reject)
        self.comboBox_resolution.currentTextChanged.connect(self.on_resolution_changed)

    def init_resolution_list(self):
        resolutions = [
            "1024 x 768",
            "1280 x 720",
            "1280 x 800",
            "1366 x 768",
            "1440 x 900",
            "1600 x 900",
            "1680 x 1050",
            "1920 x 1080",
            "2560 x 1440",
            "3840 x 2160",
            "직접 입력",
        ]
        self.comboBox_resolution.clear()
        self.comboBox_resolution.addItems(resolutions)

    def on_resolution_changed(self, text: str):
        if text == "직접 입력":
            self.custom_resolution_row.show()
            self.lineEdit_width.show()
            self.lineEdit_height.show()
            self.label_width.show()
            self.label_height.show()
        else:
            self.custom_resolution_row.hide()

    def _current_resolution_text(self) -> str | None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return None
        size = screen.size()
        return f"{size.width()} x {size.height()}"

    def load_settings(self):
        self.init_resolution_list()
        settings = self.data_manager.get_settings()
        resolution = settings.get("resolution")
        is_custom = settings.get("custom", False)

        if resolution is None:
            self.comboBox_resolution.setCurrentIndex(-1)
        elif is_custom:
            self.comboBox_resolution.setCurrentText("직접 입력")
            width, height = resolution.split("x")
            self.lineEdit_width.setText(width.strip())
            self.lineEdit_height.setText(height.strip())
            self.custom_resolution_row.show()
        else:
            index = self.comboBox_resolution.findText(resolution)
            if index >= 0:
                self.comboBox_resolution.setCurrentIndex(index)

        self.lineEdit_hotkey.setText(self.data_manager.get_capture_hotkey())
        self.lineEdit_run_hotkey.setText(self.data_manager.get_run_hotkey())
        self.lineEdit_stop_hotkey.setText(self.data_manager.get_stop_hotkey())

        current_resolution = self._current_resolution_text() or "확인 불가"
        self.label_tip.setText(f"현재 화면 해상도: {current_resolution}")
        if settings.get("setup_completed"):
            self.label_status.setText("초기 설정 완료")
        else:
            self.label_status.setText("처음 실행 전에는 해상도와 핫키를 먼저 저장해야 합니다.")

    def _validate_resolution_input(self) -> str | None:
        current_resolution = self.comboBox_resolution.currentText()
        if not current_resolution:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "해상도를 선택해 주세요.")
            return None

        if current_resolution == "직접 입력":
            width = self.lineEdit_width.text().strip()
            height = self.lineEdit_height.text().strip()
            if not width or not height:
                QtWidgets.QMessageBox.warning(self, "입력 오류", "직접 입력 해상도를 입력해 주세요.")
                return None
            return f"{width} x {height}"

        return current_resolution

    def save_settings(self):
        resolution_text = self._validate_resolution_input()
        if resolution_text is None:
            return

        self.data_manager.update_settings(
            resolution=resolution_text,
            custom=self.comboBox_resolution.currentText() == "직접 입력",
            capture_hotkey=self.lineEdit_hotkey.text().strip() or "F1",
            run_hotkey=self.lineEdit_run_hotkey.text().strip() or "F11",
            stop_hotkey=self.lineEdit_stop_hotkey.text().strip() or "F12",
            expected_resolution=self._current_resolution_text(),
            dpi_scale_locked=True,
            setup_completed=bool(self._current_resolution_text()),
        )
        self.accept()

    def reject(self):
        if self.force_setup:
            return
        super().reject()
