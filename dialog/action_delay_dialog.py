from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
from utils.path_manager import ui_path

class ActionDelayDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(str(ui_path("ActiondelayWindow.ui")), self)
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        """UI 요소 초기화"""
        self.setFixedSize(self.size())
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        # 라인에딧
        self.lineEdit_delay = self.findChild(QtWidgets.QLineEdit, 'lineEdit_delay')
        self.lineEdit_delay.setValidator(QDoubleValidator(0, 999.99, 2))
        
        # 버튼
        self.button_save = self.findChild(QtWidgets.QPushButton, 'button_save')
        self.button_cancel = self.findChild(QtWidgets.QPushButton, 'button_cancel')
        
    def connect_signals(self):
        """시그널 연결"""
        self.button_save.clicked.connect(self.save_delay)
        self.button_cancel.clicked.connect(self.close)
        
    def save_delay(self):
        """딜레이 저장"""
        try:
            delay_time = float(self.lineEdit_delay.text())
            if delay_time <= 0:
                raise ValueError
            self.accept()
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "입력 오류",
                "0보다 큰 숫자를 입력해주세요."
            )
