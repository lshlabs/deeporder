from __future__ import annotations

from pathlib import Path

from PIL import ImageGrab
from PyQt6 import QtCore, QtWidgets, uic

from core_functions.hotkey_monitor import HotkeyMonitor
from dialog.action_dialog import ActionDialog
from dialog.region_capture_dialog import RegionCaptureDialog
from utils.data_manager import DataManager
from utils.path_manager import ui_path


class ActionWizardDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title_text: str = ""):
        super().__init__(parent)
        self.title_text = title_text
        self.data_manager = DataManager.get_instance()
        self.screenshot_path: str | None = None
        self.macro_key: str | None = None
        self._waiting_for_hotkey = False
        self.hotkey_monitor = HotkeyMonitor(self.data_manager.get_capture_hotkey(), "F1")
        self._build_ui()
        self._connect_signals()

    def _refresh_hotkey(self) -> str:
        self.hotkey_monitor.update(self.data_manager.get_capture_hotkey())
        return self.hotkey_monitor.text

    def _monitor_intro_text(self) -> str:
        hotkey_text = self._refresh_hotkey()
        return (
            "감시모드 시작을 누르면 프로그램이 최소화됩니다.\n"
            f"최소화된 상태에서 {hotkey_text}를 눌러 전체 화면을 캡처합니다."
        )

    def _build_ui(self):
        uic.loadUi(str(ui_path("ActionWizardDialog.ui")), self)
        self.label_title = self.findChild(QtWidgets.QLabel, "label_title")
        self.status_label = self.findChild(QtWidgets.QLabel, "label_status")
        self.button_start = self.findChild(QtWidgets.QPushButton, "button_start")
        self.button_cancel = self.findChild(QtWidgets.QPushButton, "button_cancel")

        self.label_title.setText(self.title_text)
        self.status_label.setText(self._monitor_intro_text())

    def _connect_signals(self):
        self.button_start.clicked.connect(self.enter_monitor_mode)
        self.button_cancel.clicked.connect(self.reject)

    def enter_monitor_mode(self):
        self._waiting_for_hotkey = True
        hotkey_text = self._refresh_hotkey()
        self.button_start.setEnabled(False)
        self.button_start.setText(f"{hotkey_text} 대기 중")
        self.status_label.setText(
            "프로그램이 최소화되었습니다.\n"
            f"최소화된 상태에서 {hotkey_text}를 눌러 캡처하세요."
        )
        if self.parent():
            self.parent().showMinimized()
        self.showMinimized()
        QtCore.QTimer.singleShot(120, self._wait_for_capture_hotkey)

    def _wait_for_capture_hotkey(self):
        if not self._waiting_for_hotkey:
            return
        if self.hotkey_monitor.is_pressed():
            self._waiting_for_hotkey = False
            QtCore.QTimer.singleShot(120, self._capture_and_open_regions)
            return
        QtCore.QTimer.singleShot(50, self._wait_for_capture_hotkey)

    def _reset_wait_state(self):
        self.button_start.setEnabled(True)
        self.button_start.setText("감시모드 시작")
        self.status_label.setText(self._monitor_intro_text())

    def _capture_and_open_regions(self):
        try:
            screenshot = ImageGrab.grab(all_screens=True)
        except (OSError, ValueError) as error:
            if self.parent():
                self.parent().showNormal()
            self.showNormal()
            self._reset_wait_state()
            QtWidgets.QMessageBox.warning(self, "캡처 오류", f"화면 캡처에 실패했습니다.\n{error}")
            return

        temp_dir = Path(__file__).resolve().parents[1] / "temp"
        temp_dir.mkdir(exist_ok=True)
        self.screenshot_path = str((temp_dir / "monitor_capture.png").resolve())
        screenshot.save(self.screenshot_path)

        if self.parent():
            self.parent().showNormal()
            self.parent().raise_()
        self.showNormal()
        self.raise_()

        dialog = RegionCaptureDialog(self.screenshot_path, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dialog.capture_regions:
            self._reset_wait_state()
            return

        self.macro_key = self.data_manager.create_macro_from_capture(
            self.title_text,
            dialog.capture_regions,
            self.screenshot_path,
        )
        editor = ActionDialog(self.parent(), self.macro_key)
        editor.exec()
        self.accept()

    def reject(self):
        self._waiting_for_hotkey = False
        super().reject()
