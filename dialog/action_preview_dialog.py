from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QIcon, QIntValidator
from utils.data_manager import DataManager
from utils.path_manager import resolve_project_path, ui_path

class ActionPreviewDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, macro_key=None, action_key=None):
        super().__init__(parent)
        uic.loadUi(str(ui_path("ActionpreviewWindow.ui")), self)
        
        # 기본 설정
        self.data_manager = DataManager.get_instance()
        self.macro_key = macro_key
        self.action_key = action_key
        self.action_data = self.data_manager._data['macro_list'][macro_key]['actions'][action_key]
        
        # UI 초기화
        self.init_ui()
        self.connect_signals()
        
        # 액션 데이터 로드
        self.load_action_data()

    def init_ui(self):
        """UI 요소 초기화"""
        self.setFixedSize(self.size())
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        # 라벨
        self.label_preview = self.findChild(QtWidgets.QLabel, 'label_preview')
        self.label_X = self.findChild(QtWidgets.QLabel, 'label_X')
        self.label_Y = self.findChild(QtWidgets.QLabel, 'label_Y')
        
        # 버튼
        self.button_save = self.findChild(QtWidgets.QPushButton, 'button_save')
        self.button_cancel = self.findChild(QtWidgets.QPushButton, 'button_cancel')
        
        # 라인에딧
        self.lineEdit = self.findChild(QtWidgets.QLineEdit, 'lineEdit')
        self.lineEdit_X = self.findChild(QtWidgets.QLineEdit, 'lineEdit_X')
        self.lineEdit_Y = self.findChild(QtWidgets.QLineEdit, 'lineEdit_Y')
        
        # 체크박스
        self.checkBox_time = self.findChild(QtWidgets.QCheckBox, 'checkBox_time')
        
        # 좌표 입력 필드에 숫자만 입력되도록 설정
        self.lineEdit_X.setValidator(QIntValidator())
        self.lineEdit_Y.setValidator(QIntValidator())

    def connect_signals(self):
        """시그널 연결"""
        self.button_save.clicked.connect(self.save_action)
        self.button_cancel.clicked.connect(self.close)

    def load_action_data(self):
        """액션 데이터 로드"""
        # 이름 설정
        self.lineEdit.setText(self.action_data['name'])
        
        # 좌표 설정
        #self.lineEdit_X.setText(str(self.action_data['x']))
        #self.lineEdit_Y.setText(str(self.action_data['y']))
        
        # 이미지 설정
        if 'image' in self.action_data:
            image_path = resolve_project_path(self.action_data['image'])
            if image_path:
                pixmap = QPixmap(str(image_path))
                self.label_preview.setPixmap(pixmap.scaled(
                    self.label_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
        
        # 시간 조정 동작 여부 설정
        #if 'is_time_action' in self.action_data:
        #    self.checkBox_time.setChecked(self.action_data['is_time_action'])

    def save_action(self):
        """변경사항 저장"""
        # 라인에딧 포커스 해제하여 입력 중인 글자 확정
        self.lineEdit.clearFocus()
        
        # 입력값 검증
        if not self.lineEdit.text().strip():
            QtWidgets.QMessageBox.warning(self, "경고", "동작 이름을 입력하세요.")
            return
            
        if not self.lineEdit_X.text() or not self.lineEdit_Y.text():
            QtWidgets.QMessageBox.warning(self, "경고", "좌표값을 입력하세요.")
            return
        
        # 액션 데이터 업데이트
        self.action_data['name'] = self.lineEdit.text()
        #self.action_data['x'] = int(self.lineEdit_X.text())
        #self.action_data['y'] = int(self.lineEdit_Y.text())
        #self.action_data['is_time_action'] = self.checkBox_time.isChecked()
        
        # 데이터 저장
        self.data_manager.save_data()
        
        # 부모 다이얼로그의 테이블 업데이트
        if isinstance(self.parent(), QtWidgets.QDialog):
            self.parent().load_actions()
        
        self.accept()
