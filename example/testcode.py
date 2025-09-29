import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap
from utils.temp_manager import TempManager

class WizardStep2Dialog(QtWidgets.QDialog):
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
        
        self.init_ui()
        self.button_save.clicked.connect(self.save_and_close)
        
        # 초기 라벨(minus) 활성화
        self.handle_label_click('minus')
        
    def init_ui(self):
        self.setFixedSize(self.size())
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        self.label_preview = self.findChild(QtWidgets.QLabel, 'label_preview')
        self.labels = {name: self.findChild(QtWidgets.QLabel, f'label_{name}') 
                      for name in ['plus', 'minus', 'time']}
        
        for name, label in self.labels.items():
            label.mousePressEvent = lambda e, n=name: self.handle_label_click(n)

    def handle_label_click(self, label_type):
        # 스타일 변경만 처리
        for name, label in self.labels.items():
            style = label.styleSheet()
            if name == label_type:
                style = style.replace('#EFEFEF', 'deepskyblue').replace('black', 'white')
            else:
                style = style.replace('deepskyblue', '#EFEFEF').replace('white', 'black')
            label.setStyleSheet(style)
        self.current_drag_label = label_type
        
        # original_pixmap이 있을 때만 미리보기 업데이트
        if self.original_pixmap:
            self.update_preview()

    def handle_mouse_event(self, event, is_release=False):
        if not self.current_drag_label or not self.display_pixmap or not self.origin:
            return
            
        # label 내에서의 마우스 위치
        pos = self.label_preview.mapFromParent(event.pos())
        
        # pixmap이 그려지는 시작 위치 계산 (여백)
        x_offset = (self.label_preview.width() - self.display_pixmap.width()) // 2
        y_offset = (self.label_preview.height() - self.display_pixmap.height()) // 2
        
        # 여백을 고려한 실제 pixmap 내에서의 좌표로 변환
        pos = QPoint(
            int(pos.x() - x_offset),
            int(pos.y() - y_offset)
        )
        
        # display_pixmap 크기 내에 있는지 체크
        if not (0 <= pos.x() <= self.display_pixmap.width() and 
                0 <= pos.y() <= self.display_pixmap.height()):
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

    def mousePressEvent(self, event):
        if not self.current_drag_label or not self.display_pixmap:
            return
            
        # label 내에서의 마우스 위치
        pos = self.label_preview.mapFromParent(event.pos())
        
        # pixmap이 그려지는 시작 위치 계산 (여백)
        x_offset = (self.label_preview.width() - self.display_pixmap.width()) // 2
        y_offset = (self.label_preview.height() - self.display_pixmap.height()) // 2
        
        # 여백을 고려한 실제 pixmap 내에서의 좌표로 변환
        pos = QPoint(
            int(pos.x() - x_offset),
            int(pos.y() - y_offset)
        )
        
        # display_pixmap 크기 내에 있는지 체크
        if 0 <= pos.x() <= self.display_pixmap.width() and 0 <= pos.y() <= self.display_pixmap.height():
            self.origin = pos
            self.drag_areas[self.current_drag_label] = None
            self.current_rect = QRect(pos, pos)
            self.update_preview()

    def mouseMoveEvent(self, event):
        self.handle_mouse_event(event)
        
    def mouseReleaseEvent(self, event):
        self.handle_mouse_event(event, True)

    def save_and_close(self):
        """저장 후 창 닫기"""
        # 스케일 비율 계산 (스케일된 크기 -> 원본 크기로 변환하기 위한 비율)
        scale_x = self.original_pixmap.width() / self.display_pixmap.width()
        scale_y = self.original_pixmap.height() / self.display_pixmap.height()
        
        print(f"\n원본 이미지 크기: {self.original_pixmap.width()} x {self.original_pixmap.height()}")
        print(f"스케일된 이미지 크기: {self.display_pixmap.width()} x {self.display_pixmap.height()}")
        
        # 드래그 영역 저장 및 크롭
        for label, rect in self.drag_areas.items():
            if rect:
                # 스케일된 이미지에서의 좌표 출력
                print(f"\n{label} 영역:")
                print(f"스케일된 이미지 좌표: x:{rect.x()}, y:{rect.y()}, width:{rect.width()}, height:{rect.height()}")
                
                # 크롭을 위해 원본 크기로 변환
                original_rect = QRect(
                    int(rect.x() * scale_x),
                    int(rect.y() * scale_y),
                    int(rect.width() * scale_x),
                    int(rect.height() * scale_y)
                )
                
                # 원본 이미지에서의 좌표 출력
                print(f"원본 이미지 좌표: x:{original_rect.x()}, y:{original_rect.y()}, width:{original_rect.width()}, height:{original_rect.height()}")
                
                # 해당 영역만 크롭하여 저장 (원본 크기로)
                cropped = self.original_pixmap.copy(original_rect)
                TempManager.get_instance().save_painted_image(cropped, label)
        
        # tempdata.json에는 스케일된 좌표 저장
        TempManager.get_instance().save_drag_areas(self.drag_areas)
        
        # 전체 이미지에 드래그 영역 표시하여 저장 (원본 크기로)
        result = self.original_pixmap.copy()
        painter = QPainter(result)
        painter.setPen(QPen(QColor('red'), 1))
        for rect in self.drag_areas.values():
            if rect:
                # 그리기 전에 원본 크기로 변환
                original_rect = QRect(
                    int(rect.x() * scale_x),
                    int(rect.y() * scale_y),
                    int(rect.width() * scale_x),
                    int(rect.height() * scale_y)
                )
                painter.drawRect(original_rect)
        painter.end()
        TempManager.get_instance().save_temp_image(result, 2)
        
        self.accept()

    def update_preview(self):
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
        
        # 현재 드래그 영역만 그리기
        for area in self.drag_areas.values():
            if area:
                painter.drawRect(area)  # 스케일 변환 없이 직접 그리기
            
        if self.current_rect:
            painter.drawRect(self.current_rect)  # 스케일 변환 없이 직접 그리기
        
        painter.end()
        
        self.display_pixmap = result
        self.label_preview.setPixmap(result)

def main():
    app = QtWidgets.QApplication(sys.argv)
    dialog = WizardStep2Dialog()
    
    # 테스트를 위한 이미지 설정
    test_image = QPixmap('/Users/mac/Documents/GitHub/myGit/deeporder/example/쿠팡이츠 배달 레티나.png')  # 테스트 이미지 경로 설정 필요
    dialog.original_pixmap = test_image
    dialog.display_pixmap = test_image.copy()
    dialog.label_preview.setPixmap(dialog.display_pixmap)
    dialog.update_preview()
    
    dialog.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 