from __future__ import annotations

from pathlib import Path

from PIL import ImageGrab
from PyQt6 import QtCore, QtWidgets

from core_functions.hotkey_monitor import HotkeyMonitor
from dialog.action_dialog import ActionDialog
from dialog.region_capture_dialog import RegionCaptureDialog
from utils.data_manager import DataManager


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
        self.setWindowTitle("감시모드 진입")
        self.setFixedSize(460, 260)
        self.setStyleSheet(
            "QDialog { background: #f3f4f6; }"
            "QFrame#Card { background: white; border: 1px solid #d1d5db; border-radius: 14px; }"
            "QLabel { color: #111827; font-family: 'Malgun Gothic'; }"
            "QPushButton {"
            "  min-height: 36px;"
            "  padding: 0 14px;"
            "  border-radius: 10px;"
            "  border: 1px solid #c7ccd4;"
            "  background: #ffffff;"
            "  color: #111827;"
            "  font: 9pt 'Malgun Gothic';"
            "}"
            "QPushButton#PrimaryButton {"
            "  background: #0ea5e9;"
            "  border: 1px solid #0284c7;"
            "  color: white;"
            "  font-weight: 700;"
            "}"
            "QPushButton:disabled { background: #dbeafe; color: #6b7280; border-color: #bfdbfe; }"
        )

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)

        card = QtWidgets.QFrame(self)
        card.setObjectName("Card")
        root_layout.addWidget(card)

        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(22, 20, 22, 18)
        card_layout.setSpacing(14)

        chip = QtWidgets.QLabel("MONITOR MODE")
        chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        chip.setFixedWidth(120)
        chip.setStyleSheet(
            "QLabel { background: #e0f2fe; color: #0369a1; border-radius: 10px; padding: 6px 10px; font: 700 8pt 'Malgun Gothic'; }"
        )
        card_layout.addWidget(chip, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        title = QtWidgets.QLabel(self.title_text)
        title.setWordWrap(True)
        title.setStyleSheet("QLabel { font: 700 13pt 'Malgun Gothic'; }")
        card_layout.addWidget(title)

        self.status_label = QtWidgets.QLabel(self._monitor_intro_text())
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("QLabel { font: 9pt 'Malgun Gothic'; line-height: 1.4; }")
        card_layout.addWidget(self.status_label)

        hint = QtWidgets.QLabel("이미지 업로드 없이 바로 화면을 캡처하고, 이어서 영역 선택으로 이동합니다.")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #4b5563; font: 8pt 'Malgun Gothic'; }")
        card_layout.addWidget(hint)

        card_layout.addStretch(1)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(10)
        action_row.addStretch(1)

        self.button_cancel = QtWidgets.QPushButton("취소")
        action_row.addWidget(self.button_cancel)

        self.button_start = QtWidgets.QPushButton("감시모드 시작")
        self.button_start.setObjectName("PrimaryButton")
        action_row.addWidget(self.button_start)

        card_layout.addLayout(action_row)

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
        except Exception as error:
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
