from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from pathlib import Path
import sys

# Try to import AppKit for macOS app detection
try:
    from AppKit import NSWorkspace
    HAS_APPKIT = True
except ImportError:
    print("Warning: AppKit not available. Program detection will be limited.")
    NSWorkspace = None
    HAS_APPKIT = False
sys.path.append(str(Path(__file__).parents[1]))
from dialog.image_dialog import ImageDialog
from dialog.action_delay_dialog import ActionDelayDialog
from dialog.action_setting_dialog import ActionSettingDialog
from dialog.action_preview_dialog import ActionPreviewDialog
from utils.data_manager import DataManager

class ActionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, macro_key=""):
        super().__init__(parent)
        uic.loadUi('ui/ActionWindow.ui', self)
        self.setFixedSize(500, 570)
        self.macro_key = macro_key
        self.data_manager = DataManager.get_instance()
        self.macro_data = self.data_manager._data['macro_list'][self.macro_key]
        
        # 원본 데이터 백업
        self.original_actions = {k: v.copy() for k, v in self.macro_data['actions'].items()}
        self.original_program = self.macro_data.get('program')
        
        self.init_ui()
        self.connect_signals()
        self.load_actions()

    def init_ui(self):
        """UI 요소 초기화"""
        self._init_table()
        self._init_scroll_arrows()
        self._init_program_combo()
        self._init_buttons()
        self.label_title = self.findChild(QtWidgets.QLabel, 'label_title')
        self.label_title.setText(self.macro_data['name'])

    def _init_table(self):
        """테이블 위젯 초기화"""
        self.table_widget = self.findChild(QtWidgets.QTableWidget, 'tableWidget')
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(['번호', '이름'])
        self.table_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        
        header = self.table_widget.horizontalHeader()
        header.resizeSection(0, header.sectionSize(0) // 2)

    def _init_scroll_arrows(self):
        """스크롤 화살표 초기화"""
        self.button_arrowup = self.findChild(QtWidgets.QPushButton, 'button_arrowup')
        self.button_arrowdown = self.findChild(QtWidgets.QPushButton, 'button_arrowdown')
        self.button_arrowup.hide()
        self.button_arrowdown.hide()
        
        scrollbar = self.table_widget.verticalScrollBar()
        scrollbar.valueChanged.connect(self.update_scroll_arrows)
        scrollbar.rangeChanged.connect(self.update_scroll_arrows)

    def _init_program_combo(self):
        """프로그램 콤보박스 초기화"""
        self.combo_window = self.findChild(QtWidgets.QComboBox, 'comboBox_window')
        self.update_window_list()

    def _init_buttons(self):
        """버튼 요소 초기화"""
        button_names = ['add', 'delay', 'priority', 'edit', 'delete', 'setting', 
                       'save', 'cancel', 'up', 'down']
        for name in button_names:
            setattr(self, f'button_{name}', 
                   self.findChild(QtWidgets.QPushButton, f'button_{name}'))

    def connect_signals(self):
        """시그널 연결"""
        signals = {
            'add': self.btn_add,
            'delay': self.btn_delay,
            'priority': self.btn_priority,
            'edit': self.btn_edit,
            'delete': self.btn_delete,
            'setting': self.btn_setting,
            'save': self.btn_save,
            'cancel': self.btn_cancel,
            'up': self.btn_move_up,
            'down': self.btn_move_down
        }
        
        for name, handler in signals.items():
            getattr(self, f'button_{name}').clicked.connect(handler)
            
        self.combo_window.currentTextChanged.connect(self.on_combo_changed)

    def get_selected_action(self):
        """현재 선택된 액션의 키와 데이터 반환"""
        current_row = self.table_widget.currentRow()
        if current_row < 0:
            return None, None
            
        sorted_actions = sorted(
            self.macro_data['actions'].items(),
            key=lambda x: x[1]['number']
        )
        action_key = sorted_actions[current_row][0]
        return action_key, self.macro_data['actions'][action_key]

    def update_action_numbers(self):
        """액션 번호 재정렬"""
        sorted_actions = sorted(
            self.macro_data['actions'].items(),
            key=lambda x: x[1]['number']
        )
        for i, (key, action) in enumerate(sorted_actions):
            action['number'] = i + 1

    def save_and_reload(self):
        """데이터 저장 및 테이블 갱신"""
        self.data_manager.save_data()
        self.load_actions()

    def on_combo_changed(self, text):
        """콤보박스 선택 변경 시 실행"""
        self.macro_data['program'] = None if text == "선택 안 함" else text

    def btn_add(self):
        """추가 버튼 클릭 시 실행
        ActionWizardDialog를 생성하여 현재 매크로에 새로운 액션 5개 추가
        """
        # 동적 import로 순환 참조 해결
        from dialog.action_wizard_dialog import ActionWizardDialog
        
        dialog = ActionWizardDialog(self, title_text=self.macro_data['name'])
        dialog.macro_key = self.macro_key
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.load_actions()

    def btn_delay(self):
        """딜레이 버튼 클릭 시 실행"""
        dialog = ActionDelayDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            delay_time = dialog.lineEdit_delay.text()
            self.data_manager.create_delay_action(self.macro_key, delay_time)
            self.load_actions()

    def btn_priority(self):
        """우선순위 버튼 클릭 시 실행"""
        action_key, action_data = self.get_selected_action()
        if not action_key:
            return
            
        action_data['priority'] = not action_data.get('priority', False)
        self.save_and_reload()

    def btn_edit(self):
        """수정 버튼 클릭 시 실행"""
        action_key, action_data = self.get_selected_action()
        if not action_key:
            return
            
        if action_data['type'] == 'delay':
            dialog = ActionDelayDialog(self)
            dialog.lineEdit_delay.setText(str(action_data['value']))
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                action_data['value'] = float(dialog.lineEdit_delay.text())
                action_data['name'] = f"딜레이: {action_data['value']}초"
                self.load_actions()
        # else:
        #     dialog = ImageDialog(self, macro_key=self.macro_key, action_key=action_key)
        #     if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        #         self.load_actions()
        else:
            dialog = ActionPreviewDialog(self, macro_key=self.macro_key, action_key=action_key)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self.load_actions()

    def btn_delete(self):
        """삭제 버튼 클릭 시 실행"""
        action_key, _ = self.get_selected_action()
        if not action_key:
            return
            
        reply = QtWidgets.QMessageBox.question(
            self, '삭제 확인', '정말 삭제하시겠습니까?',
            QtWidgets.QMessageBox.StandardButton.Yes | 
            QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            del self.macro_data['actions'][action_key]
            self.update_action_numbers()
            self.save_and_reload()

    def btn_setting(self):
        """설정 버튼 클릭 시 실행"""
        dialog = ActionSettingDialog(self)
        dialog.show()

    def btn_save(self):
        """저장 버튼 클릭 시 실행"""
        self.data_manager._data['macro_list'][self.macro_key] = self.macro_data
        self.data_manager.save_data()
        self.accept()

    def btn_cancel(self):
        """취소 버튼 클릭 시 실행"""
        current_program = self.macro_data.get('program')
        if current_program == "선택 안 함":
            current_program = None
        
        has_changes = (
            self.original_actions != self.macro_data['actions'] or
            self.original_program != current_program
        )
        
        if has_changes:
            reply = QtWidgets.QMessageBox.question(
                self,
                '저장되지 않은 변경사항',
                '수정사항이 있습니다.\n저장하지 않고 종료하시겠습니까?',
                QtWidgets.QMessageBox.StandardButton.Yes | 
                QtWidgets.QMessageBox.StandardButton.No
            )
            
            if reply == QtWidgets.QMessageBox.StandardButton.No:
                return
        
            # 이미지 파일 정리 추가
            macro_folder = self.data_manager.img_path / self.macro_data['name']
            if macro_folder.exists():
                original_image_files = {Path(action['image']).name 
                                      for action in self.original_actions.values() 
                                      if action['type'] == 'image'}
                
                for file in macro_folder.glob('*.png'):
                    if file.name not in original_image_files:
                        try:
                            file.unlink()
                        except Exception as e:
                            print(f"이미지 파일 삭제 중 오류 발생: {e}")
        
        self.macro_data['actions'] = self.original_actions
        self.macro_data['program'] = self.original_program
        self.data_manager.save_data()
        self.reject()

    def btn_move_up(self):
        """위로 이동 버튼 클릭 시 실행"""
        current_row = self.table_widget.currentRow()
        if current_row <= 0:
            return
            
        sorted_actions = sorted(
            self.macro_data['actions'].items(),
            key=lambda x: x[1]['number']
        )
        
        current_key = sorted_actions[current_row][0]
        upper_key = sorted_actions[current_row - 1][0]
        
        # number 교체
        self.macro_data['actions'][current_key]['number'], \
        self.macro_data['actions'][upper_key]['number'] = \
        self.macro_data['actions'][upper_key]['number'], \
        self.macro_data['actions'][current_key]['number']
        
        self.save_and_reload()
        self.table_widget.setCurrentCell(current_row - 1, 1)

    def btn_move_down(self):
        """아래로 이동 버튼 클릭 시 실행"""
        current_row = self.table_widget.currentRow()
        if current_row >= self.table_widget.rowCount() - 1:
            return
            
        sorted_actions = sorted(
            self.macro_data['actions'].items(),
            key=lambda x: x[1]['number']
        )
        
        current_key = sorted_actions[current_row][0]
        lower_key = sorted_actions[current_row + 1][0]
        
        # number 교체
        self.macro_data['actions'][current_key]['number'], \
        self.macro_data['actions'][lower_key]['number'] = \
        self.macro_data['actions'][lower_key]['number'], \
        self.macro_data['actions'][current_key]['number']
        
        self.save_and_reload()
        self.table_widget.setCurrentCell(current_row + 1, 1)

    def update_scroll_arrows(self, direction=None):
        """스크롤바 상태에 따라 화살표 표시 업데이트 및 스크롤 처리"""
        scrollbar = self.table_widget.verticalScrollBar()
        
        # 스크롤 값 변경
        if direction == 'up':
            scrollbar.setValue(scrollbar.value() - 1)
        elif direction == 'down':
            scrollbar.setValue(scrollbar.value() + 1)
        
        # 스크롤이 필요없는 경우 화살표 숨김
        if scrollbar.maximum() == 0:
            self.button_arrowup.hide()
            self.button_arrowdown.hide()
            return
        
        # 현재 스크롤 위치에 따라 화살표 표시
        current_pos = scrollbar.value()
        max_pos = scrollbar.maximum()
        
        self.button_arrowup.setVisible(current_pos > 0)
        self.button_arrowdown.setVisible(current_pos < max_pos)

    def update_window_list(self):
        """실행 중인 프로그램 목록 업데이트"""
        try:
            exclude_list = {
                'Finder', 'Dock', 'Spotlight', 'NotificationCenter', 'ControlCenter',
                'SystemUIServer', 'loginwindow', 'WindowServer'
            }
            
            if HAS_APPKIT and NSWorkspace:
                workspace = NSWorkspace.sharedWorkspace()
                running_apps = workspace.runningApplications()
                
                filtered_programs = {
                    app.localizedName() for app in running_apps 
                    if app.activationPolicy() == 0 
                    and app.localizedName() 
                    and not app.localizedName().startswith('.')
                    and app.localizedName() not in exclude_list
                }
            else:
                # Fallback when AppKit is not available
                filtered_programs = {
                    'Google Chrome', 'Safari', 'Firefox', 'Designer', 
                    'Visual Studio Code', 'PyCharm', 'Finder'
                }
            
            self.combo_window.clear()
            self.combo_window.addItem("선택 안 함")
            self.combo_window.addItems(sorted(filtered_programs))
            
            current_program = self.macro_data.get('program')
            if current_program:
                index = self.combo_window.findText(current_program)
                if index >= 0:
                    self.combo_window.setCurrentIndex(index)
                    
        except Exception as e:
            print(f"프로그램 목록 가져오기 실패: {e}")
            self.combo_window.clear()
            self.combo_window.addItem("선택 안 함")

    def load_actions(self):
        """액션 데이터를 테이블에 로드"""
        sorted_actions = sorted(
            self.macro_data['actions'].items(),
            key=lambda x: x[1]['number']
        )
        
        self.table_widget.setRowCount(len(sorted_actions))
        
        for row, (_, action_data) in enumerate(sorted_actions):
            # 번호 열
            number_item = QtWidgets.QTableWidgetItem(str(action_data['number']))
            number_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_widget.setItem(row, 0, number_item)
            
            # 이름 열
            name = action_data['name']
            if action_data.get('priority', False):
                name += ' (우선순위 지정됨)'
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.table_widget.setItem(row, 1, name_item)
        
        if self.table_widget.rowCount() > 0:
            self.table_widget.setCurrentCell(0, 1)