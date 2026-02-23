from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap
from utils.temp_manager import TempManager
from utils.path_manager import ui_path

class WizardStep3Dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(str(ui_path("Step3Window.ui")), self)
        
        self.current_rect = None
        self.current_drag_label = None
        self.drag_areas = dict.fromkeys(['accept', 'reject'])
        self.display_pixmap = None
        self.original_pixmap = None
        self.origin = None
        
        # 저장된 드래그 영역 불러오기
        saved_areas = TempManager.get_instance().get_drag_areas()
        if saved_areas:
            for label in ['accept', 'reject']:
                if label in saved_areas and saved_areas[label]:
                    area = saved_areas[label]
                    self.drag_areas[label] = QRect(
                        area['x'], 
                        area['y'], 
                        area['width'], 
                        area['height']
                    )
        
        self.init_ui()
        self.button_save.clicked.connect(self.save_and_close)
        
        # 초기 라벨(reject) 활성화
        self.handle_label_click('reject')
        
    def init_ui(self):
        """UI 요소 초기화"""
        self.setFixedSize(self.size())
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        # 라벨
        self.label_preview = self.findChild(QtWidgets.QLabel, 'label_preview')
        self.labels = {
            'accept': self.findChild(QtWidgets.QLabel, 'label_accept'),
            'reject': self.findChild(QtWidgets.QLabel, 'label_reject')
        }
        
        # 버튼
        self.button_save = self.findChild(QtWidgets.QPushButton, 'button_save')
        
        # 클릭 이벤트 연결
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
            
        # 스케일 비율 계산
        scale_x = self.display_pixmap.width() / self.original_pixmap.width()
        scale_y = self.display_pixmap.height() / self.original_pixmap.height()
        
        # label 내에서의 마우스 위치
        pos = self.label_preview.mapFromParent(event.pos())
        
        # pixmap이 그려지는 시작 위치 계산 (여백)
        x_offset = (self.label_preview.width() - self.display_pixmap.width()) // 2
        y_offset = (self.label_preview.height() - self.display_pixmap.height()) // 2
        
        # 여백을 고려한 실제 pixmap 내에서의 좌표로 변환
        pos = QPoint(
            int((pos.x() - x_offset) / scale_x),
            int((pos.y() - y_offset) / scale_y)
        )
        
        if not self.label_preview.rect().contains(pos):
            if is_release:  # 영역 밖에서 떼면 드래그 취소
                self.current_rect = None
                self.origin = None
                self.update_preview()
            return
            
        pos.setX(max(0, min(pos.x(), self.original_pixmap.width())))
        pos.setY(max(0, min(pos.y(), self.original_pixmap.height())))
        
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
            
        # 스케일 비율 계산
        scale_x = self.display_pixmap.width() / self.original_pixmap.width()
        scale_y = self.display_pixmap.height() / self.original_pixmap.height()
        
        # label 내에서의 마우스 위치
        pos = self.label_preview.mapFromParent(event.pos())
        
        # pixmap이 그려지는 시작 위치 계산 (여백)
        x_offset = (self.label_preview.width() - self.display_pixmap.width()) // 2
        y_offset = (self.label_preview.height() - self.display_pixmap.height()) // 2
        
        # 여백을 고려한 실제 pixmap 내에서의 좌표로 변환
        pos = QPoint(
            int((pos.x() - x_offset) / scale_x),
            int((pos.y() - y_offset) / scale_y)
        )
        
        if self.label_preview.rect().contains(pos):
            self.origin = pos
            # 드래그 시작할 때 해당 라벨의 드래그 영역 초기화
            self.drag_areas[self.current_drag_label] = None
            self.current_rect = QRect(pos, pos)
            self.update_preview()

    def mouseMoveEvent(self, event):
        self.handle_mouse_event(event)
        
    def mouseReleaseEvent(self, event):
        self.handle_mouse_event(event, True)
        
    def save_and_close(self):
        """저장 후 창 닫기"""
        # 스케일된 드래그 영역 저장 (GUI 상 좌표)
        TempManager.get_instance().save_drag_areas(self.drag_areas)
        
        # 스케일 비율 계산 (원본 좌표로 변환하기 위해)
        scale_x = self.original_pixmap.width() / self.display_pixmap.width()
        scale_y = self.original_pixmap.height() / self.display_pixmap.height()
        
        # 원본 이미지 좌표로 변환된 드래그 영역 계산
        original_drag_areas = {}
        for label, rect in self.drag_areas.items():
            if rect:
                original_rect = QRect(
                    int(rect.x() * scale_x * 2),
                    int(rect.y() * scale_y * 2),
                    int(rect.width() * scale_x * 2),
                    int(rect.height() * scale_y * 2)
                )
                original_drag_areas[label] = original_rect
            else:
                original_drag_areas[label] = None
            
        # 원본 좌표 드래그 영역 저장 (tempdata의 "original_drag_areas" 키에 저장)
        TempManager.get_instance().save_original_drag_areas(original_drag_areas)
        
        # painted 이미지 저장: 원본 이미지에 해당 드래그 영역만 빨간 테두리로 그린 후 저장
        for label, rect in self.drag_areas.items():
            if rect:
                painted = self.original_pixmap.copy()
                painter = QPainter(painted)
                painter.setPen(QPen(QColor('red'), 1))
                painter.drawRect(rect)
                painter.end()
                TempManager.get_instance().save_painted_image(painted, label)
        
        # 모든 드래그 영역이 그려진 임시 이미지 저장 (temp_step3.png)
        result = self.original_pixmap.copy()
        painter = QPainter(result)
        painter.setPen(QPen(QColor('red'), 1))
        for area in self.drag_areas.values():
            if area:
                painter.drawRect(area)
        painter.end()
        TempManager.get_instance().save_temp_image(result, 3)
        
        # 부모 창의 미리보기 업데이트
        if self.parent():
            parent = self.parent()
            parent.label_preview3.setPixmap(result.scaled(
                parent.label_preview3.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # 모든 영역이 설정되었는지 확인
            all_areas_set = all(area is not None for area in self.drag_areas.values())
            if all_areas_set:
                # 부모 창의 button_save 활성화
                parent.button_save.setEnabled(True)
                parent.button_save.setStyleSheet(parent.button_save.styleSheet().replace('darkgray', '#f0f0f0'))
        
        self.accept()
        
    def update_preview(self):
        # 먼저 이미지를 label_preview 크기에 맞게 스케일링
        scaled_pixmap = self.original_pixmap.scaled(
            self.label_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # 스케일링된 이미지에 드래그 영역 그리기
        result = scaled_pixmap.copy()
        painter = QPainter(result)
        painter.setPen(QPen(QColor('red'), 1))
        
        # 스케일 비율 계산
        scale_x = scaled_pixmap.width() / self.original_pixmap.width()
        scale_y = scaled_pixmap.height() / self.original_pixmap.height()
        
        # 드래그 영역 좌표를 스케일링하여 그리기
        for area in self.drag_areas.values():
            if area:
                scaled_rect = QRect(
                    int(area.x() * scale_x),
                    int(area.y() * scale_y),
                    int(area.width() * scale_x),
                    int(area.height() * scale_y)
                )
                painter.drawRect(scaled_rect)
            
        if self.current_rect:
            scaled_rect = QRect(
                int(self.current_rect.x() * scale_x),
                int(self.current_rect.y() * scale_y),
                int(self.current_rect.width() * scale_x),
                int(self.current_rect.height() * scale_y)
            )
            painter.drawRect(scaled_rect)
        
        painter.end()
        
        self.display_pixmap = result
        self.label_preview.setPixmap(result)
