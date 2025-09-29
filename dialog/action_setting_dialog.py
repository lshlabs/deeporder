from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt

class ActionSettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('ui/ActionsettingWindow.ui', self)
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        """UI 요소 초기화"""
        self.setFixedSize(self.size())
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        # 콤보박스
        self.comboBox_run = self.findChild(QtWidgets.QComboBox, 'comboBox_run')
        self.comboBox_stop = self.findChild(QtWidgets.QComboBox, 'comboBox_stop')
        
        # 체크박스
        self.checkBox_ctrl1 = self.findChild(QtWidgets.QCheckBox, 'checkBox_ctrl1')
        self.checkBox_alt1 = self.findChild(QtWidgets.QCheckBox, 'checkBox_alt1')
        self.checkBox_shift1 = self.findChild(QtWidgets.QCheckBox, 'checkBox_shift1')
        self.checkBox_ctrl2 = self.findChild(QtWidgets.QCheckBox, 'checkBox_ctrl2')
        self.checkBox_alt2 = self.findChild(QtWidgets.QCheckBox, 'checkBox_alt2')
        self.checkBox_shift2 = self.findChild(QtWidgets.QCheckBox, 'checkBox_shift2')
        
        # 스핀박스
        self.spinBox_default = self.findChild(QtWidgets.QSpinBox, 'spinBox_default')
        self.spinBox_stack = self.findChild(QtWidgets.QSpinBox, 'spinBox_stack')
        self.spinBox_delay = self.findChild(QtWidgets.QSpinBox, 'spinBox_delay')
        
        # 버튼
        self.button_save = self.findChild(QtWidgets.QPushButton, 'button_save')
        self.button_cancel = self.findChild(QtWidgets.QPushButton, 'button_cancel')
        
    def connect_signals(self):
        """시그널 연결"""
        self.button_save.clicked.connect(self.save_settings)
        self.button_cancel.clicked.connect(self.close)
        
    def save_settings(self):
        pass