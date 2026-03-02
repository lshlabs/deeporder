from PyQt6 import QtCore, QtGui, QtWidgets, uic

from utils.data_manager import DataManager
from utils.path_manager import ui_path


class HotkeyLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setPlaceholderText("키 입력")

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()
        if key in {QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete}:
            self.clear()
            event.accept()
            return

        modifiers = event.modifiers()
        if key in {
            QtCore.Qt.Key.Key_Control,
            QtCore.Qt.Key.Key_Shift,
            QtCore.Qt.Key.Key_Alt,
            QtCore.Qt.Key.Key_Meta,
            QtCore.Qt.Key.Key_unknown,
        }:
            event.accept()
            return

        parts = []
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            parts.append("Ctrl")
        if modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
            parts.append("Alt")
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            parts.append("Shift")
        if modifiers & QtCore.Qt.KeyboardModifier.MetaModifier:
            parts.append("Meta")

        text = QtGui.QKeySequence(key).toString()
        if not text:
            text = event.text().upper().strip()
        text = (text or "").strip()
        if text:
            if len(text) == 1 or text.upper().startswith("F"):
                text = text.upper()
            parts.append(text)
            self.setText("+".join(parts))
        event.accept()


class MainSettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, force_setup: bool = False):
        super().__init__(parent)
        uic.loadUi(str(ui_path("MainsettingWindow.ui")), self)
        self.data_manager = DataManager.get_instance()
        self.force_setup = force_setup
        self.init_ui()
        self.connect_signals()
        self.load_settings()

    def init_ui(self):
        self.resize(360, 390)
        self.setFixedSize(self.size())
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

        self.frame_main.setGeometry(10, 10, 340, 370)
        self.button_save.setGeometry(160, 330, 80, 30)
        self.button_cancel.setGeometry(250, 330, 80, 30)

        self.init_resolution_list()
        self.hide_custom_input()
        self._inject_extra_controls()

        if self.force_setup:
            self.button_cancel.setEnabled(False)

    def _inject_extra_controls(self):
        frame = self.frame_main
        self.label_hotkey = QtWidgets.QLabel("캡처 핫키", frame)
        self.label_hotkey.setGeometry(30, 128, 80, 24)
        self.label_hotkey.setStyleSheet("border:none; color:black;")

        self.lineEdit_hotkey = HotkeyLineEdit(frame)
        self.lineEdit_hotkey.setGeometry(120, 126, 150, 28)
        self.lineEdit_hotkey.setStyleSheet("color:black; border-radius: 5px; padding-left:3px; background:white;")

        self.label_run_hotkey = QtWidgets.QLabel("실행 핫키", frame)
        self.label_run_hotkey.setGeometry(30, 168, 80, 24)
        self.label_run_hotkey.setStyleSheet("border:none; color:black;")

        self.lineEdit_run_hotkey = HotkeyLineEdit(frame)
        self.lineEdit_run_hotkey.setGeometry(120, 166, 150, 28)
        self.lineEdit_run_hotkey.setStyleSheet("color:black; border-radius: 5px; padding-left:3px; background:white;")

        self.label_stop_hotkey = QtWidgets.QLabel("종료 핫키", frame)
        self.label_stop_hotkey.setGeometry(30, 208, 80, 24)
        self.label_stop_hotkey.setStyleSheet("border:none; color:black;")

        self.lineEdit_stop_hotkey = HotkeyLineEdit(frame)
        self.lineEdit_stop_hotkey.setGeometry(120, 206, 150, 28)
        self.lineEdit_stop_hotkey.setStyleSheet("color:black; border-radius: 5px; padding-left:3px; background:white;")

        self.label_status = QtWidgets.QLabel(frame)
        self.label_status.setGeometry(30, 252, 280, 52)
        self.label_status.setWordWrap(True)
        self.label_status.setStyleSheet("border:none; color:black;")

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

    def connect_signals(self):
        self.button_save.clicked.connect(self.save_settings)
        self.button_cancel.clicked.connect(self.reject)
        self.comboBox_resolution.currentTextChanged.connect(self.on_resolution_changed)

    def hide_custom_input(self):
        self.lineEdit_width.hide()
        self.lineEdit_height.hide()
        self.label_width.hide()
        self.label_height.hide()

    def show_custom_input(self):
        self.lineEdit_width.show()
        self.lineEdit_height.show()
        self.label_width.show()
        self.label_height.show()

    def on_resolution_changed(self, text):
        if text == "직접 입력":
            self.show_custom_input()
        else:
            self.hide_custom_input()

    def _current_resolution_text(self):
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return None
        size = screen.size()
        return f"{size.width()} x {size.height()}"

    def load_settings(self):
        settings = self.data_manager._data["settings_main"]
        resolution = settings.get("resolution")
        is_custom = settings.get("custom", False)

        if resolution is None:
            self.comboBox_resolution.setCurrentIndex(-1)
        elif is_custom:
            self.comboBox_resolution.setCurrentText("직접 입력")
            width, height = resolution.split("x")
            self.lineEdit_width.setText(width.strip())
            self.lineEdit_height.setText(height.strip())
            self.show_custom_input()
        else:
            index = self.comboBox_resolution.findText(resolution)
            if index >= 0:
                self.comboBox_resolution.setCurrentIndex(index)

        self.lineEdit_hotkey.setText(settings.get("capture_hotkey", "F1"))
        self.lineEdit_run_hotkey.setText(settings.get("run_hotkey", "F11"))
        self.lineEdit_stop_hotkey.setText(settings.get("stop_hotkey", "F12"))
        current_resolution = self._current_resolution_text() or "확인 불가"
        setup_done = settings.get("setup_completed", False)
        self.label_tip.setText(f"현재 화면 해상도: {current_resolution}")
        self.label_status.setText(
            "초기 설정 완료" if setup_done else "최초 실행 전 안정 동작을 위해 해상도와 핫키를 저장해야 합니다."
        )

    def validate_custom_resolution(self):
        if not self.lineEdit_width.text() or not self.lineEdit_height.text():
            QtWidgets.QMessageBox.warning(self, "입력 오류", "해상도를 입력해주세요.")
            return False
        return True

    def save_settings(self):
        current_resolution = self.comboBox_resolution.currentText()
        hotkey = self.lineEdit_hotkey.text().strip() or "F1"
        run_hotkey = self.lineEdit_run_hotkey.text().strip() or "F11"
        stop_hotkey = self.lineEdit_stop_hotkey.text().strip() or "F12"
        if not current_resolution:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "해상도를 선택해주세요.")
            return

        settings = self.data_manager._data["settings_main"]
        if current_resolution == "직접 입력":
            if not self.validate_custom_resolution():
                return
            current_resolution = f"{self.lineEdit_width.text()} x {self.lineEdit_height.text()}"
            settings["custom"] = True
        else:
            settings["custom"] = False

        settings["resolution"] = current_resolution
        settings["capture_hotkey"] = hotkey
        settings["run_hotkey"] = run_hotkey
        settings["stop_hotkey"] = stop_hotkey
        settings["expected_resolution"] = self._current_resolution_text()
        settings["dpi_scale_locked"] = True
        settings["setup_completed"] = bool(settings["expected_resolution"])

        self.data_manager.save_data()
        self.accept()

    def reject(self):
        if self.force_setup:
            return
        super().reject()
