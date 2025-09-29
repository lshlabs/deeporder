import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PyQt6.QtWidgets import QLabel, QDialog
from utils.temp_manager import TempManager
from area_detection_logic import AreaDetectionLogic

class AreaSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('deeporder/ui/Step2Window.ui', self)
        
        # 임시 파일들 초기화
        TempManager.get_instance().clear_temp_data()
        
        # 필요한 변수들 초기화
        self.current_rect = None
        self.current_drag_label = None
        self.drag_areas = dict.fromkeys(['plus', 'minus', 'time'])
        self.display_pixmap = None
        self.original_pixmap = None
        self.origin = None
        
        # 로직 클래스 인스턴스 생성
        self.logic = AreaDetectionLogic()
        
        self.init_ui()
        self.connect_signals()
        
        # 초기 라벨(minus) 활성화
        self.handle_label_click('minus')
        
    def init_ui(self):
        # 창 설정
        self.setFixedSize(self.size())
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        # UI 요소 참조
        self.label_preview = self.findChild(QtWidgets.QLabel, 'label_preview')
        self.labels = {name: self.findChild(QtWidgets.QLabel, f'label_{name}') 
                      for name in ['plus', 'minus', 'time']}
        self.button_save = self.findChild(QtWidgets.QPushButton, 'button_save')
    
    def connect_signals(self):
        # 버튼 이벤트 연결
        self.button_save.clicked.connect(self.save_and_close)
        
        # 라벨 클릭 이벤트 연결
        for name, label in self.labels.items():
            label.mousePressEvent = lambda e, n=name: self.handle_label_click(n)

    def handle_label_click(self, label_type):
        # 스타일 변경 처리
        for name, label in self.labels.items():
            style = label.styleSheet()
            if name == label_type:
                style = style.replace('#EFEFEF', 'deepskyblue').replace('black', 'white')
            else:
                style = style.replace('deepskyblue', '#EFEFEF').replace('white', 'black')
            label.setStyleSheet(style)
        
        self.current_drag_label = label_type
        
        # 미리보기 업데이트
        if self.original_pixmap:
            self.update_preview()

    def mousePressEvent(self, event):
        if not self.current_drag_label or not self.display_pixmap:
            return
        
        # 로직 클래스에 위임
        pos = self.logic.calculate_image_coordinates(
            event.pos(), 
            self.label_preview, 
            self.display_pixmap
        )
        
        if pos and self.logic.is_within_image(pos, self.display_pixmap):
            self.origin = pos
            self.drag_areas[self.current_drag_label] = None
            self.current_rect = QRect(pos, pos)
            self.update_preview()

    def mouseMoveEvent(self, event):
        self.handle_mouse_event(event)
        
    def mouseReleaseEvent(self, event):
        self.handle_mouse_event(event, True)

    def handle_mouse_event(self, event, is_release=False):
        if not self.current_drag_label or not self.display_pixmap or not self.origin:
            return
        
        # 로직 클래스에 위임
        pos = self.logic.calculate_image_coordinates(
            event.pos(), 
            self.label_preview, 
            self.display_pixmap
        )
        
        if not pos or not self.logic.is_within_image(pos, self.display_pixmap):
            if is_release:
                self.current_rect = None
                self.origin = None
                self.update_preview()
            return
        
        if is_release:
            if self.current_rect and self.current_rect.width() > 5 and self.current_rect.height() > 5:
                # 유효한 크기의 드래그만 저장
                self.drag_areas[self.current_drag_label] = QRect(self.origin, pos).normalized()
            self.current_rect = None
            self.origin = None
        else:
            self.current_rect = QRect(self.origin, pos).normalized()
        
        self.update_preview()

    def save_and_close(self):
        """저장 후 스크린샷 프로세스 시작"""
        if not self.original_pixmap or not self.display_pixmap:
            self.accept()
            return
        
        # 로직 클래스에 처리 위임
        self.logic.save_selected_areas(
            self.original_pixmap, 
            self.display_pixmap, 
            self.drag_areas
        )
        
        # 스크린샷 프로세스 시작 - 대화 상자는 아직 닫지 않음
        self.logic.start_screenshot_process(self, self.drag_areas)
        
        # 대화 상자는 스크린샷 프로세스가 완료된 후 닫힘
        # self.accept()는 스크린샷 프로세스가 완료된 후 호출됨
    
    def finish_process_and_close(self):
        """스크린샷 프로세스 완료 후 대화 상자 닫기"""
        self.accept()

    def update_preview(self):
        """미리보기 업데이트"""
        if not self.original_pixmap:
            return
            
        # 먼저 이미지를 label_preview 크기에 맞게 스케일링
        scaled_pixmap = self.original_pixmap.scaled(
            self.label_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # 새로운 이미지에 드래그 영역 그리기
        result = scaled_pixmap.copy()
        painter = QPainter(result)
        painter.setPen(QPen(QColor('red'), 1))
        
        # 저장된 드래그 영역 그리기
        for area in self.drag_areas.values():
            if area:
                painter.drawRect(area)
                
        # 현재 드래그 중인 영역 그리기
        if self.current_rect:
            painter.drawRect(self.current_rect)
        
        painter.end()
        
        self.display_pixmap = result
        self.label_preview.setPixmap(result)

    def set_image(self, pixmap):
        """이미지 설정"""
        if pixmap:
            self.original_pixmap = pixmap
            self.display_pixmap = pixmap.copy()
            self.label_preview.setPixmap(self.display_pixmap)
            self.update_preview()

def main():
    app = QtWidgets.QApplication(sys.argv)
    dialog = AreaSelectionDialog()
    
    # 테스트를 위한 이미지 설정
    test_image = QPixmap('/Users/mac/Documents/GitHub/myGit/deeporder/example/쿠팡이츠 배달 레티나.png')
    dialog.set_image(test_image)
    
    dialog.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()