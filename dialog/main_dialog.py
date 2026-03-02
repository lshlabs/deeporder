import ctypes
import ctypes.wintypes
import logging
import os

from PyQt6 import QtCore, QtWidgets, uic
from PyQt6.QtGui import QKeySequence, QShortcut

from core_functions.macro_runner import MacroRunner
from dialog.action_dialog import ActionDialog
from dialog.action_wizard_dialog import ActionWizardDialog
from dialog.main_setting_dialog import MainSettingDialog
from utils.data_manager import DataManager
from utils.hotkey_utils import parse_hotkey_for_register
from utils.logger_ui import bind_text_widget
from utils.path_manager import ui_path


RUNNING_SUFFIX = " (실행 중)"


WM_HOTKEY = 0x0312
HOTKEY_ID_RUN = 1
HOTKEY_ID_STOP = 2
class GlobalHotkeyEventFilter(QtCore.QAbstractNativeEventFilter):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def nativeEventFilter(self, eventType, message):
        if eventType != "windows_generic_MSG":
            return False, 0
        try:
            msg = ctypes.wintypes.MSG.from_address(int(message))
        except Exception:
            return False, 0
        if msg.message == WM_HOTKEY:
            self.callback(int(msg.wParam))
            return True, 0
        return False, 0


class MainDialog(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(str(ui_path("MainWindow.ui")), self)
        self.setFixedSize(500, 700)

        self.macro_runner = MacroRunner()
        self.macro_runner.status_changed.connect(self.on_macro_status_changed)
        self.macro_runner.log_message.connect(self.on_log_message)
        self.data_manager = DataManager.get_instance()
        self.macro_name_to_key = {}
        self.shortcut_run_selected = None
        self.shortcut_stop_all = None
        self.global_hotkey_filter = None
        self._registered_hotkey_ids = set()

        self.init_ui()
        self.init_log_ui()
        self.init_shortcuts()
        self.connect_signals()
        self.load_macro_list()
        QtCore.QTimer.singleShot(0, self.ensure_setup_ready)

    def init_ui(self):
        self.label_title = self.findChild(QtWidgets.QLabel, "label_title")
        self.label_run = self.findChild(QtWidgets.QLabel, "label_run")
        self.label_stop = self.findChild(QtWidgets.QLabel, "label_stop")
        self.listWidget = self.findChild(QtWidgets.QListWidget, "listWidget")
        self.lineEdit = self.findChild(QtWidgets.QLineEdit, "lineEdit")
        self.button_add = self.findChild(QtWidgets.QPushButton, "button_add")
        self.button_delete = self.findChild(QtWidgets.QPushButton, "button_delete")
        self.button_edit = self.findChild(QtWidgets.QPushButton, "button_edit")
        self.button_copy = self.findChild(QtWidgets.QPushButton, "button_copy")
        self.button_setting = self.findChild(QtWidgets.QPushButton, "button_setting")
        self.textBrowser_log = self.findChild(QtWidgets.QPlainTextEdit, "textBrowser_log")
        self.button_add.setText("생성")
        self._apply_compact_fonts()

    def _apply_compact_fonts(self):
        exempt = {self.label_title, self.label_run, self.label_stop}
        for widget in self.findChildren(QtWidgets.QWidget):
            if widget in exempt:
                continue
            font = widget.font()
            font.setFamily("Malgun Gothic")
            if isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QLineEdit, QtWidgets.QListWidget, QtWidgets.QPlainTextEdit)):
                font.setPointSize(9)
            elif isinstance(widget, (QtWidgets.QLabel, QtWidgets.QComboBox)):
                font.setPointSize(9)
            widget.setFont(font)

    def init_log_ui(self):
        self.ui_logger = logging.getLogger("deeporder.runtime")
        self.ui_logger.setLevel(logging.INFO)
        self.ui_logger.propagate = False
        if self.textBrowser_log:
            self.textBrowser_log.setReadOnly(True)
            self.log_handler = bind_text_widget(self.textBrowser_log, logger_name="deeporder.runtime", max_lines=500)
            self.ui_logger.handlers = [self.log_handler]

    def init_shortcuts(self):
        self._ensure_global_hotkey_filter()
        self.refresh_hotkeys()

    def init_tray_icon(self):
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = None
            return

        self.tray_icon = QtWidgets.QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.windowIcon() or self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon))
        self.tray_icon.setToolTip("DeepOrder")

        tray_menu = QtWidgets.QMenu(self)
        action_open = tray_menu.addAction("열기")
        action_stop = tray_menu.addAction("모든 매크로 중지")
        tray_menu.addSeparator()
        action_exit = tray_menu.addAction("종료")

        action_open.triggered.connect(self.show_from_tray)
        action_stop.triggered.connect(self.stop_all_running_macros)
        action_exit.triggered.connect(self.exit_application)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def _ensure_global_hotkey_filter(self):
        if os.name != "nt":
            return
        app = QtWidgets.QApplication.instance()
        if app is None or self.global_hotkey_filter is not None:
            return
        self.global_hotkey_filter = GlobalHotkeyEventFilter(self.on_global_hotkey)
        app.installNativeEventFilter(self.global_hotkey_filter)

    def _unregister_global_hotkeys(self):
        if os.name != "nt":
            return
        user32 = ctypes.windll.user32
        for hotkey_id in list(self._registered_hotkey_ids):
            try:
                user32.UnregisterHotKey(None, hotkey_id)
            except Exception:
                pass
        self._registered_hotkey_ids.clear()

    def _register_global_hotkey(self, hotkey_id: int, hotkey_text: str, fallback: str):
        if os.name != "nt":
            return False
        modifiers, key_code = parse_hotkey_for_register(hotkey_text, fallback)
        try:
            registered = bool(ctypes.windll.user32.RegisterHotKey(None, hotkey_id, modifiers, key_code))
        except Exception:
            registered = False
        if registered:
            self._registered_hotkey_ids.add(hotkey_id)
        return registered

    def refresh_hotkeys(self):
        run_hotkey = self.data_manager.get_run_hotkey()
        stop_hotkey = self.data_manager.get_stop_hotkey()

        if self.shortcut_run_selected is not None:
            self.shortcut_run_selected.activated.disconnect()
            self.shortcut_run_selected.setParent(None)
            self.shortcut_run_selected = None
        if self.shortcut_stop_all is not None:
            self.shortcut_stop_all.activated.disconnect()
            self.shortcut_stop_all.setParent(None)
            self.shortcut_stop_all = None

        self._unregister_global_hotkeys()

        if os.name == "nt":
            run_ok = self._register_global_hotkey(HOTKEY_ID_RUN, run_hotkey, "F11")
            stop_ok = self._register_global_hotkey(HOTKEY_ID_STOP, stop_hotkey, "F12")
            if not run_ok:
                self.on_log_message(f"실행 핫키 등록 실패: {run_hotkey}")
            if not stop_ok:
                self.on_log_message(f"종료 핫키 등록 실패: {stop_hotkey}")
            return

        self.shortcut_run_selected = QShortcut(QKeySequence(run_hotkey), self)
        self.shortcut_run_selected.activated.connect(self.start_selected_macro)

        self.shortcut_stop_all = QShortcut(QKeySequence(stop_hotkey), self)
        self.shortcut_stop_all.activated.connect(self.stop_all_running_macros)

    def on_global_hotkey(self, hotkey_id: int):
        if hotkey_id == HOTKEY_ID_RUN:
            self.start_selected_macro()
        elif hotkey_id == HOTKEY_ID_STOP:
            self.stop_all_running_macros()

    def connect_signals(self):
        self.button_add.clicked.connect(self.btn_add)
        self.button_delete.clicked.connect(self.btn_delete)
        self.button_edit.clicked.connect(self.btn_edit)
        self.button_copy.clicked.connect(self.btn_copy)
        self.button_setting.clicked.connect(self.btn_setting)
        self.label_run.mousePressEvent = self.label_run_clicked
        self.label_stop.mousePressEvent = self.label_stop_clicked
        self.listWidget.itemClicked.connect(self.listWidget_item_clicked)

    def _strip_running_suffix(self, text: str):
        return text.replace(RUNNING_SUFFIX, "")

    def _is_running_item_text(self, text: str):
        return str(text).endswith(RUNNING_SUFFIX)

    def ensure_setup_ready(self):
        settings = self.data_manager.get_settings()
        current_resolution = self._current_resolution_text()
        needs_setup = not settings.get("setup_completed") or (
            settings.get("expected_resolution") and settings.get("expected_resolution") != current_resolution
        )
        if needs_setup:
            dialog = MainSettingDialog(self, force_setup=True)
            dialog.exec()
            self.data_manager = DataManager.get_instance()
            self.refresh_hotkeys()

    def _current_resolution_text(self):
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return None
        size = screen.size()
        return f"{size.width()} x {size.height()}"

    def btn_add(self):
        if not self.validate_lineEdit():
            return
        if not self.data_manager.is_setup_completed():
            self.ensure_setup_ready()
            if not self.data_manager.is_setup_completed():
                return

        self.button_add.setFocus()
        QtWidgets.QApplication.processEvents()
        title_text = self.lineEdit.text().strip()
        dialog = ActionWizardDialog(self, title_text=title_text)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.lineEdit.clear()
            self.load_macro_list()

    def btn_delete(self):
        current_item = self.listWidget.currentItem()
        if not current_item:
            return
        if self._is_running_item_text(current_item.text()):
            QtWidgets.QMessageBox.warning(self, "삭제 오류", "실행 중인 매크로는 삭제할 수 없습니다.")
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "삭제 확인",
            "정말 삭제하시겠습니까?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        macro_name = self._strip_running_suffix(current_item.text())
        macro_key = self.macro_name_to_key.get(macro_name)
        if macro_key:
            self.data_manager.delete_macro(macro_key)
        self.load_macro_list()

    def btn_edit(self):
        current_item = self.listWidget.currentItem()
        if not current_item:
            return
        macro_name = self._strip_running_suffix(current_item.text())
        macro_key = self.macro_name_to_key.get(macro_name)
        if not macro_key:
            return
        dialog = ActionDialog(self, macro_key)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.load_macro_list()

    def btn_copy(self):
        current_item = self.listWidget.currentItem()
        if not current_item:
            return
        base_text = self._strip_running_suffix(current_item.text())
        copy_num = 1
        while True:
            new_text = f"{base_text} ({copy_num})"
            if new_text not in self.macro_name_to_key:
                break
            copy_num += 1
        original_macro_key = self.macro_name_to_key.get(base_text)
        if original_macro_key:
            self.data_manager.copy_macro(original_macro_key, new_text)
            self.load_macro_list()

    def btn_setting(self):
        dialog = MainSettingDialog(self)
        dialog.exec()
        self.data_manager = DataManager.get_instance()
        self.refresh_hotkeys()

    def stop_all_running_macros(self):
        self.macro_runner.stop_all()
        stop_hotkey = self.data_manager.get_stop_hotkey()
        self.on_log_message(f"{stop_hotkey} 긴급 중지: 실행 중 매크로 중지 요청")
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.load_macro_list()

    def start_selected_macro(self):
        try:
            current_item = self.listWidget.currentItem()
            if not current_item:
                return
            macro_name = self._strip_running_suffix(current_item.text())
            macro_key = self.macro_name_to_key.get(macro_name)
            if not macro_key:
                return
            if self.macro_runner.start_macro(macro_key):
                current_item.setText(f"{macro_name}{RUNNING_SUFFIX}")
                self.showMinimized()
        except Exception as e:
            self.on_log_message(f"RUN 실행 오류: {e}")
            QtWidgets.QMessageBox.warning(self, "실행 오류", f"매크로 실행 시작에 실패했습니다.\n{e}")

    def label_run_clicked(self, event):
        self.start_selected_macro()

    def label_stop_clicked(self, event):
        current_item = self.listWidget.currentItem()
        if not current_item:
            return
        macro_name = self._strip_running_suffix(current_item.text())
        macro_key = self.macro_name_to_key.get(macro_name)
        if macro_key:
            self.macro_runner.stop_macro(macro_key)
        self.load_macro_list()

    def listWidget_item_clicked(self, item):
        return

    def validate_lineEdit(self):
        text = self.lineEdit.text().strip()
        if not text:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "이름을 입력해주세요.")
            return False
        import re

        if not re.match(r"^[a-zA-Z0-9가-힣\s]+$", text):
            QtWidgets.QMessageBox.warning(self, "입력 오류", "특수문자는 사용할 수 없습니다.")
            return False
        if text in self.macro_name_to_key:
            QtWidgets.QMessageBox.warning(self, "중복 오류", "이미 존재하는 이름입니다.")
            return False
        return True

    def on_macro_status_changed(self, macro_key, status):
        if status == "stopped":
            self.showNormal()
            self.raise_()
            self.activateWindow()
        self.load_macro_list()

    def on_log_message(self, message):
        self.ui_logger.info(message)
        print(f"로그: {message}")

    def load_macro_list(self):
        self.listWidget.clear()
        self.macro_name_to_key.clear()
        for key, macro in self.data_manager.get_macro_list().items():
            macro_name = macro.get("name", key)
            self.listWidget.addItem(macro_name)
            self.macro_name_to_key[macro_name] = key
        if self.listWidget.count() > 0:
            self.listWidget.setCurrentRow(0)

    def _minimize_to_tray(self):
        if not getattr(self, "tray_icon", None):
            return
        self.hide()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized)
        if not self._tray_hint_shown:
            self.tray_icon.showMessage(
                "DeepOrder",
                "시스템 트레이에서 실행 중입니다. 설정한 핫키로 실행/종료할 수 있습니다.",
                QtWidgets.QSystemTrayIcon.MessageIcon.Information,
                2500,
            )
            self._tray_hint_shown = True

    def show_from_tray(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def on_tray_activated(self, reason):
        if reason in {
            QtWidgets.QSystemTrayIcon.ActivationReason.Trigger,
            QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            self.show_from_tray()

    def exit_application(self):
        self._unregister_global_hotkeys()
        self.macro_runner.stop_all()
        if getattr(self, "tray_icon", None):
            self.tray_icon.hide()
        QtWidgets.QApplication.quit()

    def changeEvent(self, event):
        super().changeEvent(event)

    def closeEvent(self, event):
        self._unregister_global_hotkeys()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(
        "QWidget { font-family: 'Malgun Gothic'; font-size: 9pt; }"
        "QLabel#label_title, QLabel#label_run, QLabel#label_stop { font-size: 20pt; font-weight: 700; }"
    )
    window = MainDialog()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
