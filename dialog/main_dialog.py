import logging
import shutil
import sys
from pathlib import Path

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut

sys.path.append(str(Path(__file__).parents[1]))

from core_functions.macro_runner import MacroRunner
from dialog.action_dialog import ActionDialog
from dialog.action_wizard_dialog import ActionWizardDialog
from dialog.main_setting_dialog import MainSettingDialog
from utils.data_manager import DataManager
from utils.logger_ui import bind_text_widget
from utils.path_manager import ui_path


RUNNING_SUFFIX = " (실행 중)"


class MainDialog(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(str(ui_path("MainWindow.ui")), self)
        self.setFixedSize(500, 700)

        self.macro_runner = MacroRunner()
        self.macro_runner.status_changed.connect(self.on_macro_status_changed)
        self.macro_runner.log_message.connect(self.on_log_message)

        # listWidget 에 보이는 이름 -> 실제 macro key (M1, M2 ...)
        self.macro_name_to_key = {}

        self.init_ui()
        self.init_log_ui()
        self.init_shortcuts()
        self.connect_signals()
        self.load_macro_list()

    def init_ui(self):
        """UI 위젯 참조를 잡는 함수"""
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

    def init_log_ui(self):
        """GUI 로그창과 logging 핸들러 연결"""
        self.ui_logger = logging.getLogger("deeporder.runtime")
        self.ui_logger.setLevel(logging.INFO)
        self.ui_logger.propagate = False
        self.log_handler = None

        if self.textBrowser_log:
            self.textBrowser_log.setReadOnly(True)
            self.log_handler = bind_text_widget(
                self.textBrowser_log,
                logger_name="deeporder.runtime",
                max_lines=500,
            )
            self.ui_logger.handlers = [self.log_handler]

    def init_shortcuts(self):
        self.shortcut_stop_all = QShortcut(QKeySequence("F12"), self)
        self.shortcut_stop_all.activated.connect(self.stop_all_running_macros)

    def connect_signals(self):
        """버튼/라벨/리스트 시그널 연결"""
        self.button_add.clicked.connect(self.btn_add)
        self.button_delete.clicked.connect(self.btn_delete)
        self.button_edit.clicked.connect(self.btn_edit)
        self.button_copy.clicked.connect(self.btn_copy)
        self.button_setting.clicked.connect(self.btn_setting)

        self.label_run.mousePressEvent = self.label_run_clicked
        self.label_stop.mousePressEvent = self.label_stop_clicked
        self.listWidget.itemClicked.connect(self.listWidget_item_clicked)

    def _strip_running_suffix(self, text: str) -> str:
        return text.replace(RUNNING_SUFFIX, "")

    def _is_running_item_text(self, text: str) -> bool:
        return str(text).endswith(RUNNING_SUFFIX)

    def _find_macro_key_by_name(self, macro_name: str):
        return self.macro_name_to_key.get(macro_name)

    def _set_run_stop_label_state(self, is_running: bool):
        """라벨 색상만 간단히 바꾸는 함수 (기존 스타일 문자열 재사용)"""
        if is_running:
            run_style = self.label_run.styleSheet().replace("darkgray", "deepskyblue")
            stop_style = self.label_stop.styleSheet().replace("deepskyblue", "darkgray")
        else:
            run_style = self.label_run.styleSheet().replace("deepskyblue", "darkgray")
            stop_style = self.label_stop.styleSheet().replace("darkgray", "deepskyblue")

        self.label_run.setStyleSheet(run_style)
        self.label_stop.setStyleSheet(stop_style)

    def _refresh_title_running_style(self):
        has_running_item = False
        for i in range(self.listWidget.count()):
            if self._is_running_item_text(self.listWidget.item(i).text()):
                has_running_item = True
                break

        if has_running_item:
            title_style = self.label_title.styleSheet().replace("darkgray", "deepskyblue")
        else:
            title_style = self.label_title.styleSheet().replace("deepskyblue", "darkgray")
        self.label_title.setStyleSheet(title_style)

    def btn_add(self):
        """새 매크로 추가 (wizard 시작)"""
        if not self.validate_lineEdit():
            return

        # IME 입력 중인 상태에서 바로 dialog 열면 텍스트가 덜 반영되는 경우가 있어서 포커스 이동
        self.button_add.setFocus()
        QtWidgets.QApplication.processEvents()

        title_text = self.lineEdit.text().strip()
        dialog = ActionWizardDialog(self, title_text=title_text)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            text = self.lineEdit.text().strip()
            self.listWidget.addItem(text)
            self.lineEdit.clear()
            self.listWidget.setCurrentRow(self.listWidget.count() - 1)
            self.load_macro_list()

    def btn_delete(self):
        """선택한 매크로 삭제"""
        current_item = self.listWidget.currentItem()
        if not current_item:
            return

        if self._is_running_item_text(current_item.text()):
            QtWidgets.QMessageBox.warning(
                self,
                "삭제 오류",
                "실행 중인 매크로는 삭제할 수 없습니다.",
            )
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
        data_manager = DataManager.get_instance()
        macro_list = data_manager._data["macro_list"]

        macro_key = None
        for key, macro in macro_list.items():
            if macro.get("name") == macro_name:
                macro_key = key
                break

        if macro_key:
            macro_folder = data_manager.img_path / macro_name
            if macro_folder.exists():
                shutil.rmtree(macro_folder)

            del macro_list[macro_key]
            data_manager.save_data()

        self.listWidget.takeItem(self.listWidget.currentRow())
        self.macro_name_to_key.pop(macro_name, None)
        self._refresh_title_running_style()

    def btn_edit(self):
        """선택한 매크로 편집"""
        current_item = self.listWidget.currentItem()
        if not current_item:
            return

        if self._is_running_item_text(current_item.text()):
            QtWidgets.QMessageBox.warning(
                self,
                "편집 오류",
                "실행 중인 매크로는 편집할 수 없습니다.",
            )
            return

        current_name = self._strip_running_suffix(current_item.text())

        data_manager = DataManager.get_instance()
        macro_key = None
        for key, macro in data_manager._data["macro_list"].items():
            if macro.get("name") == current_name:
                macro_key = key
                break

        if not macro_key:
            return

        dialog = ActionDialog(self, macro_key)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_name = data_manager._data["macro_list"][macro_key]["name"]
            if new_name != current_name:
                current_item.setText(new_name)
                self.load_macro_list()

    def btn_copy(self):
        """선택한 매크로 복제"""
        current_item = self.listWidget.currentItem()
        if not current_item:
            return

        if self._is_running_item_text(current_item.text()):
            QtWidgets.QMessageBox.warning(
                self,
                "복제 오류",
                "실행 중인 매크로는 복제할 수 없습니다.",
            )
            return

        base_text = self._strip_running_suffix(current_item.text())

        copy_num = 1
        while True:
            new_text = f"{base_text} ({copy_num})"
            exists = False
            for i in range(self.listWidget.count()):
                if self.listWidget.item(i).text() == new_text:
                    exists = True
                    break
            if not exists:
                break
            copy_num += 1

        data_manager = DataManager.get_instance()
        original_macro_key = None
        for key, macro in data_manager._data["macro_list"].items():
            if macro.get("name") == base_text:
                original_macro_key = key
                break

        if original_macro_key:
            new_macro_key = data_manager.copy_macro(original_macro_key, new_text)
            if new_macro_key:
                self.listWidget.addItem(new_text)
                self.listWidget.setCurrentItem(current_item)
                self.load_macro_list()

    def btn_setting(self):
        """설정 창 열기"""
        dialog = MainSettingDialog(self)
        dialog.show()

    def stop_all_running_macros(self):
        self.macro_runner.stop_all()
        self.on_log_message("F12 긴급 중지: 실행 중 매크로 중지 요청")

        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            item.setText(self._strip_running_suffix(item.text()))

        self._set_run_stop_label_state(is_running=False)
        self._refresh_title_running_style()

    def label_run_clicked(self, event):
        """실행 라벨 클릭 -> 현재 선택 매크로 실행"""
        if self.listWidget.count() == 0:
            QtWidgets.QMessageBox.warning(self, "실행 오류", "실행할 항목이 없습니다.")
            return

        current_item = self.listWidget.currentItem()
        if not current_item:
            return

        if self._is_running_item_text(current_item.text()):
            return

        macro_name = self._strip_running_suffix(current_item.text())
        macro_key = self._find_macro_key_by_name(macro_name)

        if not macro_key:
            QtWidgets.QMessageBox.warning(
                self,
                "실행 오류",
                "매크로 정보를 찾을 수 없습니다.",
            )
            return

        success = self.macro_runner.start_macro(macro_key)
        if success:
            current_item.setText(f"{macro_name}{RUNNING_SUFFIX}")
            self._set_run_stop_label_state(is_running=True)
            self._refresh_title_running_style()

    def label_stop_clicked(self, event):
        """중지 라벨 클릭 -> 현재 선택 매크로 중지"""
        current_item = self.listWidget.currentItem()
        if current_item and self._is_running_item_text(current_item.text()):
            macro_name = self._strip_running_suffix(current_item.text())
            macro_key = self._find_macro_key_by_name(macro_name)
            if macro_key:
                self.macro_runner.stop_macro(macro_key)
            current_item.setText(macro_name)

        self._set_run_stop_label_state(is_running=False)
        self._refresh_title_running_style()

    def listWidget_item_clicked(self, item):
        """리스트 아이템 선택 시 run/stop 라벨 상태만 맞춰줌"""
        self._set_run_stop_label_state(is_running=self._is_running_item_text(item.text()))

    def manage_listWidget(self, action: str):
        """기존 코드 호환용 분기 함수"""
        if action == "add":
            self.btn_add()
        elif action == "delete":
            self.btn_delete()
        elif action == "copy":
            self.btn_copy()
        elif action == "edit":
            self.btn_edit()

    def validate_lineEdit(self) -> bool:
        """매크로 이름 입력값 검증"""
        text = self.lineEdit.text().strip()

        if not text:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "이름을 입력해주세요.")
            return False

        import re

        # 한글/영문/숫자/공백만 허용 (포트폴리오용으로 단순 규칙)
        if not re.match(r"^[a-zA-Z0-9가-힣\s]+$", text):
            QtWidgets.QMessageBox.warning(
                self,
                "입력 오류",
                "특수문자는 사용할 수 없습니다.",
            )
            return False

        for i in range(self.listWidget.count()):
            if self._strip_running_suffix(self.listWidget.item(i).text()) == text:
                QtWidgets.QMessageBox.warning(
                    self,
                    "중복 오류",
                    "이미 존재하는 이름입니다.",
                )
                return False

        return True

    def on_macro_status_changed(self, macro_key, status):
        """매크로 상태 변경 시 리스트 텍스트 동기화"""
        data_manager = DataManager.get_instance()
        macro_name = None

        if macro_key in data_manager._data["macro_list"]:
            macro_name = data_manager._data["macro_list"][macro_key]["name"]

        if not macro_name:
            return

        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            item_name = self._strip_running_suffix(item.text())

            if item_name != macro_name:
                continue

            if status == "running" and not self._is_running_item_text(item.text()):
                item.setText(f"{item_name}{RUNNING_SUFFIX}")
            elif status == "stopped" and self._is_running_item_text(item.text()):
                item.setText(item_name)
            break

        self._refresh_title_running_style()

    def on_log_message(self, message):
        """MacroRunner 로그를 GUI와 콘솔에 같이 출력"""
        if hasattr(self, "ui_logger"):
            self.ui_logger.info(message)
        print(f"로그: {message}")

    def load_macro_list(self):
        """data.json의 macro_list를 listWidget에 로드"""
        data_manager = DataManager.get_instance()
        macro_list = data_manager._data["macro_list"]

        self.listWidget.clear()
        self.macro_name_to_key.clear()

        for key, macro in macro_list.items():
            macro_name = macro.get("name", key)
            self.listWidget.addItem(macro_name)
            self.macro_name_to_key[macro_name] = key

        if self.listWidget.count() > 0:
            self.listWidget.setCurrentRow(0)

        self._refresh_title_running_style()


def main():
    app = QtWidgets.QApplication([])
    window = MainDialog()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
