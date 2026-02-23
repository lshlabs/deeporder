import logging
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from pathlib import Path
import sys
import shutil
sys.path.append(str(Path(__file__).parents[1]))
from dialog.action_wizard_dialog import ActionWizardDialog
from dialog.action_dialog import ActionDialog
from dialog.main_setting_dialog import MainSettingDialog
from utils.data_manager import DataManager
from utils.logger_ui import bind_text_widget
from utils.path_manager import ui_path
from core_functions.macro_runner import MacroRunner

class MainDialog(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(str(ui_path("MainWindow.ui")), self)
        self.setFixedSize(500, 700)
        
        # MacroRunner 인스턴스 생성
        self.macro_runner = MacroRunner()
        self.macro_runner.status_changed.connect(self.on_macro_status_changed)
        self.macro_runner.log_message.connect(self.on_log_message)
        
        # 매크로 키 매핑 (이름 -> 키)
        self.macro_name_to_key = {}
        
        self.init_ui()
        self.init_log_ui()
        self.init_shortcuts()
        self.connect_signals()
        self.load_macro_list()

    def init_ui(self):
        """UI 요소 초기화"""
        # 라벨들
        self.label_title = self.findChild(QtWidgets.QLabel, 'label_title')
        self.label_run = self.findChild(QtWidgets.QLabel, 'label_run')
        self.label_stop = self.findChild(QtWidgets.QLabel, 'label_stop')
        
        # 리스트 위젯과 라인 에딧
        self.listWidget = self.findChild(QtWidgets.QListWidget, 'listWidget')
        self.lineEdit = self.findChild(QtWidgets.QLineEdit, 'lineEdit')
        
        # 버튼들
        self.button_add = self.findChild(QtWidgets.QPushButton, 'button_add')
        self.button_delete = self.findChild(QtWidgets.QPushButton, 'button_delete')
        self.button_edit = self.findChild(QtWidgets.QPushButton, 'button_edit')
        self.button_copy = self.findChild(QtWidgets.QPushButton, 'button_copy')
        self.button_setting = self.findChild(QtWidgets.QPushButton, 'button_setting')
        self.textBrowser_log = self.findChild(QtWidgets.QPlainTextEdit, 'textBrowser_log')

    def init_log_ui(self):
        self.ui_logger = logging.getLogger("deeporder.runtime")
        self.ui_logger.setLevel(logging.INFO)
        self.ui_logger.propagate = False
        self.log_handler = None
        if self.textBrowser_log:
            self.textBrowser_log.setReadOnly(True)
            self.log_handler = bind_text_widget(self.textBrowser_log, logger_name="deeporder.runtime", max_lines=500)
            self.ui_logger.handlers = [self.log_handler]

    def init_shortcuts(self):
        self.shortcut_stop_all = QShortcut(QKeySequence("F12"), self)
        self.shortcut_stop_all.activated.connect(self.stop_all_running_macros)

    def connect_signals(self):
        """시그널 연결"""
        self.button_add.clicked.connect(self.btn_add)
        self.button_delete.clicked.connect(self.btn_delete)
        self.button_edit.clicked.connect(self.btn_edit)
        self.button_copy.clicked.connect(self.btn_copy)
        self.button_setting.clicked.connect(self.btn_setting)
        self.label_run.mousePressEvent = self.label_run_clicked
        self.label_stop.mousePressEvent = self.label_stop_clicked
        self.listWidget.itemClicked.connect(self.listWidget_item_clicked)

    def btn_add(self):
        """추가 버튼 클릭 시 실행"""
        if not self.validate_lineEdit():
            return
        
        # 포커스를 다른 위젯으로 이동시켜 IME 입력 완료를 강제
        self.button_add.setFocus()
        QtWidgets.QApplication.processEvents()
        
        title_text = self.lineEdit.text().strip()
        dialog = ActionWizardDialog(self, title_text=title_text)
        
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:  # save 버튼으로 닫힌 경우
            text = self.lineEdit.text().strip()
            self.listWidget.addItem(text)
            self.lineEdit.clear()
            # 추가된 항목 선택
            self.listWidget.setCurrentRow(self.listWidget.count() - 1)

    def btn_delete(self):
        """삭제 버튼 클릭 시 실행"""
        current_item = self.listWidget.currentItem()
        if current_item:
            # 실행 중인 아이템 삭제 시도 시 경고
            if current_item.text().endswith(' (실행 중)'):
                QtWidgets.QMessageBox.warning(
                    self,
                    "삭제 오류",
                    "실행중인 동작은 삭제할 수 없습니다."
                )
                return
            
            # 삭제 확인 메시지
            reply = QtWidgets.QMessageBox.question(
                self,
                '삭제 확인',
                '정말 삭제하시겠습니까?',
                QtWidgets.QMessageBox.StandardButton.Yes | 
                QtWidgets.QMessageBox.StandardButton.No
            )
            
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                # 데이터에서 해당 매크로 찾기
                data_manager = DataManager.get_instance()
                macro_list = data_manager._data['macro_list']
                
                # 매크로 이름으로 매크로 키 찾기
                macro_key = None
                for key, macro in macro_list.items():
                    if macro['name'] == current_item.text():
                        macro_key = key
                        break
                
                if macro_key:
                    # 이미지 폴더 삭제
                    macro_folder = data_manager.img_path / current_item.text()
                    if macro_folder.exists():
                        shutil.rmtree(macro_folder)
                    
                    # 데이터에서 매크로 삭제
                    del macro_list[macro_key]
                    data_manager.save_data()
                
                # listWidget에서 아이템 삭제
                self.listWidget.takeItem(self.listWidget.currentRow())

    def btn_edit(self):
        """편집 버튼 클릭 시 실행"""
        current_item = self.listWidget.currentItem()
        if current_item:
            # 실행 중인 아이템 편집 시도 시 경고
            if current_item.text().endswith(' (실행 중)'):
                QtWidgets.QMessageBox.warning(
                    self,
                    "편집 오류",
                    "실행중인 동작은 편집할 수 없습니다."
                )
                return
            
            # 데이터 매니저에서 매크로 키 찾기
            data_manager = DataManager.get_instance()
            macro_list = data_manager._data['macro_list']
            macro_key = None
            for key, macro in macro_list.items():
                if macro['name'] == current_item.text():
                    macro_key = key
                    break
            
            if macro_key:
                dialog = ActionDialog(self, macro_key)
                if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:  # save 버튼으로 닫힌 경우
                    # 이름이 변경된 경우 listWidget 항목 업데이트
                    new_name = data_manager._data['macro_list'][macro_key]['name']
                    if new_name != current_item.text():
                        current_item.setText(new_name)

    def btn_copy(self):
        """복제 버튼 클릭 시 실행"""
        current_item = self.listWidget.currentItem()
        if current_item:
            # 실행 중인 아이템 복제 시도 시 경고
            if current_item.text().endswith(' (실행 중)'):
                QtWidgets.QMessageBox.warning(
                    self,
                    "복제 오류",
                    "실행중인 동작은 복제할 수 없습니다."
                )
                return
            
            base_text = current_item.text()
            # 복제 번호 확인
            copy_num = 1
            while True:
                new_text = f"{base_text} ({copy_num})"
                # 같은 번호가 있는지 확인
                exists = False
                for i in range(self.listWidget.count()):
                    if self.listWidget.item(i).text() == new_text:
                        exists = True
                        break
                if not exists:
                    break
                copy_num += 1
            
            # 데이터 매니저에서 원본 매크로 키 찾기
            data_manager = DataManager.get_instance()
            macro_list = data_manager._data['macro_list']
            original_macro_key = None
            for key, macro in macro_list.items():
                if macro['name'] == base_text:
                    original_macro_key = key
                    break
            
            if original_macro_key:
                # 매크로 복제
                new_macro_key = data_manager.copy_macro(original_macro_key, new_text)
                if new_macro_key:
                    # 새 항목 추가
                    self.listWidget.addItem(new_text)
                    # 원본 항목 선택 유지
                    self.listWidget.setCurrentItem(current_item)

    def btn_setting(self):
        """설정 버튼 클릭 시 실행"""
        dialog = MainSettingDialog(self)
        dialog.show()

    def stop_all_running_macros(self):
        self.macro_runner.stop_all()
        self.on_log_message("F12 긴급 중단: 실행 중 매크로 중지 요청")

    def label_run_clicked(self, event):
        """실행 라벨 클릭 시 실행"""
        # 리스트가 비어있는 경우
        if self.listWidget.count() == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "실행 오류",
                "실행 할 동작이 없습니다."
            )
            return
        
        current_item = self.listWidget.currentItem()
        if current_item:
            # 이미 실행 중인 경우 무시
            if current_item.text().endswith(' (실행 중)'):
                return
                
            # 매크로 이름에서 키 가져오기
            macro_name = current_item.text()
            macro_key = self.macro_name_to_key.get(macro_name)
            
            if not macro_key:
                QtWidgets.QMessageBox.warning(
                    self,
                    "실행 오류",
                    "매크로 정보를 찾을 수 없습니다."
                )
                return
                
            # 매크로 실행
            success = self.macro_runner.start_macro(macro_key)
            
            if success:
                current_item.setText(f"{macro_name} (실행 중)")
                
                # 라벨들의 스타일시트 변경 (run 상태)
                run_style = self.label_run.styleSheet().replace('darkgray', 'deepskyblue')
                stop_style = self.label_stop.styleSheet().replace('deepskyblue', 'darkgray')
                
                self.label_run.setStyleSheet(run_style)
                self.label_stop.setStyleSheet(stop_style)
                
                # label_title은 실행 중인 아이템이 하나라도 있으면 run 상태 유지
                title_style = self.label_title.styleSheet().replace('darkgray', 'deepskyblue')
                self.label_title.setStyleSheet(title_style)

    def label_stop_clicked(self, event):
        """중지 라벨 클릭 시 실행"""
        current_item = self.listWidget.currentItem()
        if current_item and current_item.text().endswith(' (실행 중)'):
            macro_name = current_item.text().replace(' (실행 중)', '')
            macro_key = self.macro_name_to_key.get(macro_name)
            
            if macro_key:
                # 매크로 중지
                self.macro_runner.stop_macro(macro_key)
                
            current_item.setText(macro_name)
        
        # 라벨들의 스타일시트 변경 (stop 상태)
        run_style = self.label_run.styleSheet().replace('deepskyblue', 'darkgray')
        stop_style = self.label_stop.styleSheet().replace('darkgray', 'deepskyblue')
        
        self.label_run.setStyleSheet(run_style)
        self.label_stop.setStyleSheet(stop_style)
        
        # 실행 중인 아이템이 하나라도 있는지 확인
        has_running_item = False
        for i in range(self.listWidget.count()):
            if self.listWidget.item(i).text().endswith(' (실행 중)'):
                has_running_item = True
                break
        
        # 실행 중인 아이템 유무에 따라 label_title 상태 변경
        if has_running_item:
            title_style = self.label_title.styleSheet().replace('darkgray', 'deepskyblue')
        else:
            title_style = self.label_title.styleSheet().replace('deepskyblue', 'darkgray')
        self.label_title.setStyleSheet(title_style)

    def listWidget_item_clicked(self, item):
        """리스트 위젯 아이템 클릭 시 실행"""
        # 클릭된 아이템이 '실행 중' 상태인 경우 run 상태로, 아닌 경우 stop 상태로 변경
        if item.text().endswith(' (실행 중)'):
            # run 상태로 변경
            run_style = self.label_run.styleSheet().replace('darkgray', 'deepskyblue')
            stop_style = self.label_stop.styleSheet().replace('deepskyblue', 'darkgray')
        else:
            # stop 상태로 변경
            run_style = self.label_run.styleSheet().replace('deepskyblue', 'darkgray')
            stop_style = self.label_stop.styleSheet().replace('darkgray', 'deepskyblue')
        
        self.label_run.setStyleSheet(run_style)
        self.label_stop.setStyleSheet(stop_style)

    def manage_listWidget(self, action: str):
        """ListWidget 항목 관리"""
        if action == 'add':
            self.btn_add()
        elif action == 'delete':
            self.btn_delete()
        elif action == 'copy':
            self.btn_copy()
        elif action == 'edit':
            self.btn_edit()

    def validate_lineEdit(self) -> bool:
        """LineEdit 텍스트 유효성 검사
        
        Returns:
            bool: 유효성 검사 통과 여부
        """
        text = self.lineEdit.text().strip()
        
        # 빈 텍스트 검사
        if not text:
            QtWidgets.QMessageBox.warning(
                self,
                "입력 오류",
                "이름을 입력해주세요."
            )
            return False
        
        # 특수문자 검사 (알파벳, 숫자, 한글(자음/모음 포함), 공백만 허용)
        import re
        if not re.match(r'^[a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s]+$', text):
            QtWidgets.QMessageBox.warning(
                self,
                "입력 오류",
                "특수문자는 사용할 수 없습니다."
            )
            return False
        
        # 중복 항목 검사
        for i in range(self.listWidget.count()):
            if self.listWidget.item(i).text() == text:
                QtWidgets.QMessageBox.warning(
                    self,
                    "중복 오류",
                    "이미 존재하는 이름입니다."
                )
                return False
        
        return True

    def on_macro_status_changed(self, macro_key, status):
        """매크로 상태 변경 시그널 핸들러"""
        # 데이터에서 매크로 이름 찾기
        data_manager = DataManager.get_instance()
        macro_name = None
        if macro_key in data_manager._data['macro_list']:
            macro_name = data_manager._data['macro_list'][macro_key]['name']
        
        if not macro_name:
            return
            
        # UI 업데이트
        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            item_name = item.text().replace(' (실행 중)', '')
            
            if item_name == macro_name:
                if status == "running" and not item.text().endswith(' (실행 중)'):
                    item.setText(f"{item_name} (실행 중)")
                elif status == "stopped" and item.text().endswith(' (실행 중)'):
                    item.setText(item_name)
                break

    def on_log_message(self, message):
        """로그 메시지 시그널 핸들러"""
        self.ui_logger.info(message)
        print(f"로그: {message}")

    def load_macro_list(self):
        """데이터에서 매크로 리스트 로드"""
        data_manager = DataManager.get_instance()
        macro_list = data_manager._data['macro_list']
        
        # listWidget 초기화
        self.listWidget.clear()
        self.macro_name_to_key.clear()  # 매핑 초기화
        
        # 매크로 이름들을 listWidget에 추가
        for key, macro in macro_list.items():
            self.listWidget.addItem(macro['name'])
            # 이름 -> 키 매핑 추가
            self.macro_name_to_key[macro['name']] = key
        
        # 첫 번째 아이템이 있다면 선택
        if self.listWidget.count() > 0:
            self.listWidget.setCurrentRow(0)

def main():
    app = QtWidgets.QApplication([])
    window = MainDialog()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()
