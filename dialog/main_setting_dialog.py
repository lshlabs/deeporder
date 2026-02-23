from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from utils.data_manager import DataManager
from utils.path_manager import ui_path

class MainSettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(str(ui_path("MainsettingWindow.ui")), self)
        self.data_manager = DataManager.get_instance()
        self.init_ui()
        self.connect_signals()
        self.load_settings()
        
    def init_ui(self):
        """UI 요소 초기화"""
        self.setFixedSize(self.size())
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        # 콤보박스
        self.comboBox_resolution = self.findChild(QtWidgets.QComboBox, 'comboBox_run')
        
        # 라인에딧
        self.lineEdit_width = self.findChild(QtWidgets.QLineEdit, 'lineEdit_width')
        self.lineEdit_height = self.findChild(QtWidgets.QLineEdit, 'lineEdit_height')
        
        # 라벨
        self.label_width = self.findChild(QtWidgets.QLabel, 'label_width')
        self.label_height = self.findChild(QtWidgets.QLabel, 'label_height')
        
        # 버튼
        self.button_save = self.findChild(QtWidgets.QPushButton, 'button_save')
        self.button_cancel = self.findChild(QtWidgets.QPushButton, 'button_cancel')
        
        # 해상도 목록 추가
        self.init_resolution_list()
        
        # 위젯 숨기기 (초기 상태)
        self.hide_custom_input()

    def init_resolution_list(self):
        """해상도 목록 초기화"""
        resolutions = [
            "1024 x 768",    # 기본 해상도
            "1280 x 720",    # HD
            "1280 x 800",    # WXGA
            "1366 x 768",    # HD
            "1440 x 900",    # WXGA+
            "1600 x 900",    # HD+
            "1680 x 1050",   # WSXGA+
            "1920 x 1080",   # Full HD
            "2560 x 1440",   # QHD
            "3840 x 2160",   # 4K UHD
            "직접 입력"      # 직접 입력 옵션
        ]
        self.comboBox_resolution.addItems(resolutions)

    def connect_signals(self):
        """시그널 연결"""
        self.button_save.clicked.connect(self.save_settings)
        self.button_cancel.clicked.connect(self.close)
        self.comboBox_resolution.currentTextChanged.connect(self.on_resolution_changed)

    def hide_custom_input(self):
        """커스텀 입력 위젯 숨기기"""
        self.lineEdit_width.hide()
        self.lineEdit_height.hide()
        self.label_width.hide()
        self.label_height.hide()

    def show_custom_input(self):
        """커스텀 입력 위젯 보이기"""
        self.lineEdit_width.show()
        self.lineEdit_height.show()
        self.label_width.show()
        self.label_height.show()
        
    def on_resolution_changed(self, text):
        """해상도 콤보박스 선택 변경 시 실행"""
        if text == "직접 입력":
            self.show_custom_input()
        else:
            self.hide_custom_input()
        
    def load_settings(self):
        """설정 데이터 로드"""
        settings = self.data_manager._data['settings_main']
        resolution = settings.get('resolution')
        is_custom = settings.get('custom', False)
        
        if resolution is None:
            # resolution이 null인 경우 아무것도 선택하지 않음
            self.comboBox_resolution.setCurrentIndex(-1)
            return
        
        if is_custom:
            # 직접 입력한 해상도인 경우
            self.comboBox_resolution.setCurrentText("직접 입력")
            width, height = resolution.split("x")
            self.lineEdit_width.setText(width.strip())
            self.lineEdit_height.setText(height.strip())
            self.show_custom_input()
        else:
            # 콤보박스에서 선택한 해상도인 경우
            index = self.comboBox_resolution.findText(resolution)
            if index >= 0:
                self.comboBox_resolution.setCurrentIndex(index)

    def save_settings(self):
        """설정 저장"""
        current_resolution = self.comboBox_resolution.currentText()
        settings = self.data_manager._data['settings_main']
        
        if current_resolution == "직접 입력":
            if not self.validate_custom_resolution():
                return
            
            # 직접 입력한 해상도 저장
            width = self.lineEdit_width.text()
            height = self.lineEdit_height.text()
            settings['resolution'] = f"{width} x {height}"
            settings['custom'] = True
        else:
            # 콤보박스에서 선택한 해상도 저장
            settings['resolution'] = current_resolution
            settings['custom'] = False
        
        # 데이터 저장
        self.data_manager.save_data()
        self.accept()

    def validate_custom_resolution(self) -> bool:
        """커스텀 해상도 입력값 검증
        
        Returns:
            bool: 유효성 검사 통과 여부
        """
        width = self.lineEdit_width.text()
        height = self.lineEdit_height.text()
        
        if not width or not height:
            QtWidgets.QMessageBox.warning(
                self,
                "입력 오류",
                "해상도를 입력해주세요."
            )
            return False
            
        return True
