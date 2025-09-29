from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QIcon
from pathlib import Path
from utils.data_manager import DataManager

class ImageDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, macro_key=None, action_key=None):
        super().__init__(parent)
        uic.loadUi('ui/ImageWindow.ui', self)
        
        # 기본 설정
        self.data_manager = DataManager.get_instance()
        self.macro_key = macro_key
        self.action_key = action_key
        
        # 선택된 이미지 경로를 저장할 변수 추가
        self.file_path = None
        
        # UI 초기화
        self.init_ui()
        self.connect_signals()
        
        # 수정 모드
        if self.action_key is not None:
            self.action_data = self.data_manager._data['macro_list'][macro_key]['actions'][action_key]
            self.load_action_data()
        # 추가 모드
        else:
            self.label_preview.setText("이곳을 클릭하여\n이미지를 선택하세요")

    def init_ui(self):
        """UI 요소 초기화"""
        self.setFixedSize(self.size())
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        # 라벨
        self.label_preview = self.findChild(QtWidgets.QLabel, 'label_preview')
        self.label_tip = self.findChild(QtWidgets.QLabel, 'label_tip')
        self.label_surface1 = self.findChild(QtWidgets.QLabel, 'label_surface1')
        self.label_surface2 = self.findChild(QtWidgets.QLabel, 'label_surface2')
        self.label_surface3 = self.findChild(QtWidgets.QLabel, 'label_surface3')
        self.label_surface4 = self.findChild(QtWidgets.QLabel, 'label_surface4')
        
        # 버튼
        self.button_save = self.findChild(QtWidgets.QPushButton, 'button_save')
        self.button_cancel = self.findChild(QtWidgets.QPushButton, 'button_cancel')
        self.button_reset = self.findChild(QtWidgets.QPushButton, 'button_reset')
        self.button_reset.setIcon(QIcon.fromTheme('view-refresh'))
        
        # 라인에딧
        self.lineEdit = self.findChild(QtWidgets.QLineEdit, 'lineEdit')

    def connect_signals(self):
        """시그널 연결"""
        self.button_save.clicked.connect(self.save_action)
        self.button_cancel.clicked.connect(self.close)
        self.button_reset.clicked.connect(self.reset_image)
        self.label_preview.mousePressEvent = self.label_preview_clicked
        
        # surface 라벨 클릭 이벤트 연결
        for i in range(1, 5):
            label = getattr(self, f'label_surface{i}')
            label.mousePressEvent = lambda e, num=i: self.surface_label_clicked(num)

    def load_action_data(self):
        """수정 모드: 기존 액션 데이터 로드"""
        # 이름 설정
        self.lineEdit.setText(self.action_data['name'])
        
        # 이미지 설정
        image_path = self.action_data['image']
        if Path(image_path).exists():
            pixmap = QPixmap(image_path)
            self.label_preview.setPixmap(pixmap.scaled(
                self.label_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        
        # surface 설정
        surfaces = self.action_data['surface']
        
        for i in range(1, 5):
            label = getattr(self, f'label_surface{i}')
            # 모든 surface를 먼저 darkgray로 초기화
            label.setStyleSheet(label.styleSheet().replace('deepskyblue', 'darkgray'))
            # 데이터에 있는 surface만 deepskyblue로 설정
            if i in surfaces:
                label.setStyleSheet(label.styleSheet().replace('darkgray', 'deepskyblue'))

    def surface_label_clicked(self, number):
        """surface 라벨 클릭 시 처리"""
        label = getattr(self, f'label_surface{number}')
        style = label.styleSheet()
        if 'deepskyblue' in style:
            label.setStyleSheet(style.replace('deepskyblue', 'darkgray'))
        else:
            label.setStyleSheet(style.replace('darkgray', 'deepskyblue'))

    def label_preview_clicked(self, event):
        """이미지 선택"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("이미지 선택")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        
        if file_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.file_path = file_dialog.selectedFiles()[0]  # 파일 경로 저장
            pixmap = QPixmap(self.file_path)
            self.label_preview.setPixmap(pixmap.scaled(
                self.label_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def save_action(self):
        """변경사항 저장"""
        # 라인에딧 포커스 해제하여 입력 중인 글자 확정
        self.lineEdit.clearFocus()
        
        # 이미지 체크
        if not self.label_preview.pixmap():
            QtWidgets.QMessageBox.warning(
                self,
                '이미지 없음',
                '이미지를 선택해주세요.'
            )
            return
        
        # surface 정보 수집
        surfaces = []
        for i in range(1, 5):
            label = getattr(self, f'label_surface{i}')
            if 'deepskyblue' in label.styleSheet():
                surfaces.append(i)
        
        # 수정 모드
        if self.action_key is not None:
            # 이름 업데이트
            self.action_data['name'] = self.lineEdit.text()
            # 이미지 업데이트
            if self.file_path:  # 새로운 이미지가 선택된 경우
                new_image_path = str(Path(self.file_path))
                pixmap = QPixmap(self.file_path)
                pixmap.save(new_image_path)
            else:  # 기존 이미지를 유지하는 경우
                new_image_path = str(Path(self.action_data['image']))
                self.label_preview.pixmap().save(new_image_path)
            # surface 업데이트
            self.action_data['surface'] = surfaces
        
        # 추가 모드
        else:
            # 새로운 action 생성
            self.data_manager.create_image_action(
                macro_key=self.macro_key,
                name=self.lineEdit.text(),
                image_path=str(self.file_path),  # 저장된 파일 경로 사용
                surfaces=surfaces
            )
        
        # 데이터 저장
        self.data_manager.save_data()
        
        # 부모 다이얼로그(ActionDialog)의 테이블 업데이트
        if isinstance(self.parent(), QtWidgets.QDialog):
            self.parent().load_actions()
        
        self.accept()

    def reset_image(self):
        """이미지 초기화"""
        self.label_preview.setPixmap(QPixmap())
        self.label_preview.setText("이곳을 클릭하여\n이미지를 선택하세요")