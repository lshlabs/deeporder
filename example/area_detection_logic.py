import cv2
import numpy as np
import mss
import time
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt6.QtWidgets import QLabel, QDialog, QVBoxLayout, QApplication
from utils.temp_manager import TempManager

class AreaDetectionLogic:
    def __init__(self):
        self.countdown = 5
        self.timer = None
        self.countdown_dialog = None
        self.parent_dialog = None
        # 템플릿 매칭 임계값 추가
        self.TEMPLATE_MATCHING_THRESHOLD = 0.7
    
    def calculate_image_coordinates(self, screen_pos, label, pixmap):
        """화면 좌표를 이미지 내부 좌표로 변환"""
        # label 내에서의 마우스 위치
        pos = label.mapFromParent(screen_pos)
        
        # pixmap이 그려지는 시작 위치 계산 (여백)
        x_offset = (label.width() - pixmap.width()) // 2
        y_offset = (label.height() - pixmap.height()) // 2
        
        # 여백을 고려한 실제 pixmap 내에서의 좌표로 변환
        return QPoint(
            int(pos.x() - x_offset),
            int(pos.y() - y_offset)
        )
    
    def is_within_image(self, pos, pixmap):
        """좌표가 이미지 내부에 있는지 확인"""
        return (0 <= pos.x() <= pixmap.width() and 
                0 <= pos.y() <= pixmap.height())
    
    def save_selected_areas(self, original_pixmap, display_pixmap, drag_areas):
        """선택된 영역 저장"""
        # 스케일 비율 계산
        scale_x = original_pixmap.width() / display_pixmap.width()
        scale_y = original_pixmap.height() / display_pixmap.height()
        
        # 드래그 영역 저장 및 크롭
        for label, rect in drag_areas.items():
            if rect:
                # 크롭을 위해 원본 크기로 변환
                original_rect = QRect(
                    int(rect.x() * scale_x),
                    int(rect.y() * scale_y),
                    int(rect.width() * scale_x),
                    int(rect.height() * scale_y)
                )
                
                # 해당 영역만 크롭하여 저장 (원본 크기로)
                cropped = original_pixmap.copy(original_rect)
                TempManager.get_instance().save_painted_image(cropped, label)
        
        # tempdata.json에는 스케일된 좌표 저장
        TempManager.get_instance().save_drag_areas(drag_areas)
        
        # 전체 이미지에 드래그 영역 표시하여 저장 (원본 크기로)
        result = original_pixmap.copy()
        painter = QPainter(result)
        painter.setPen(QPen(QColor('red'), 1))
        
        for rect in drag_areas.values():
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
    
    def start_screenshot_process(self, parent_dialog, drag_areas):
        """스크린샷 프로세스 시작"""
        self.parent_dialog = parent_dialog
        self.drag_areas = drag_areas
        self.original_pixmap = parent_dialog.original_pixmap
        
        # 카운트다운 다이얼로그 생성
        self.countdown = 5
        self.countdown_dialog = QDialog(parent_dialog)
        self.countdown_dialog.setWindowTitle("스크린샷 준비")
        self.countdown_label = QLabel(f"{self.countdown}초 후에 스크린샷이 촬영됩니다.", self.countdown_dialog)
        self.countdown_dialog.setLayout(QVBoxLayout())
        self.countdown_dialog.layout().addWidget(self.countdown_label)
        
        # 타이머 설정 - 부모를 전달하지 않음
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)
        
        self.countdown_dialog.show()
    
    def update_countdown(self):
        """카운트다운 업데이트"""
        self.countdown -= 1
        self.countdown_label.setText(f"{self.countdown}초 후에 스크린샷이 촬영됩니다.")
        
        if self.countdown <= 0:
            self.timer.stop()
            self.countdown_dialog.close()
            self.take_screenshot()
    
    def take_screenshot(self):
        """스크린샷 캡처 및 이미지 인식"""
        # 전체 화면 스크린샷 촬영
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # 주 모니터
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 템플릿 매칭을 위해 원본 이미지를 CV2 형식으로 변환
            template = self.convert_qpixmap_to_cv2(self.original_pixmap)
            
            # 템플릿 매칭 수행
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 임계값 검사 추가
            if max_val < self.TEMPLATE_MATCHING_THRESHOLD:
                # 매칭 실패 처리
                self.show_matching_failed(img, max_val)
                return
            
            # 매칭된 위치 (기준점)
            template_x, template_y = max_loc
            
            # 스케일 비율 계산
            display_pixmap = self.parent_dialog.display_pixmap
            scale_x = self.original_pixmap.width() / display_pixmap.width()
            scale_y = self.original_pixmap.height() / display_pixmap.height()
            
            # 각 영역에 대해 중심 좌표 계산 및 시각화
            center_points = {}
            
            for label, rect in self.drag_areas.items():
                if rect:
                    # 원본 이미지 좌표로 변환
                    original_rect = QRect(
                        int(rect.x() * scale_x),
                        int(rect.y() * scale_y),
                        int(rect.width() * scale_x),
                        int(rect.height() * scale_y)
                    )
                    
                    # 기준점을 기준으로 좌표 계산
                    x = template_x + original_rect.x()
                    y = template_y + original_rect.y()
                    w = original_rect.width()
                    h = original_rect.height()
                    
                    # 중심점 계산
                    center_x = x + w // 2
                    center_y = y + h // 2
                    center_points[label] = (center_x, center_y)
                    
                    # 시각화
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 템플릿 영역 전체 표시
            h, w = template.shape[:2]
            cv2.rectangle(img, (template_x, template_y), 
                         (template_x + w, template_y + h), (0, 255, 0), 2)
            
            # 결과 이미지 저장
            cv2.imwrite('screenshot_result.png', img)
            
            # 화면 표시를 위해 RGB로 변환
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 결과 이미지 표시 - 유사도 점수 전달
            self.show_result(img_rgb, center_points, max_val)
    
    def convert_qpixmap_to_cv2(self, pixmap):
        """QPixmap을 OpenCV 이미지로 변환"""
        qimage = pixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    
    def show_result(self, cv2_img, center_points, similarity=None):
        """결과 이미지 표시"""
        height, width = cv2_img.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(cv2_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 독립적인 대화 상자 생성 (부모 없음)
        result_dialog = QDialog()
        result_dialog.setWindowTitle("스크린샷 결과")
        
        # 화면 크기의 절반으로 조정
        screen = QApplication.primaryScreen().geometry()
        dialog_width = screen.width() // 2
        dialog_height = screen.height() // 2
        result_dialog.resize(dialog_width, dialog_height)
        
        # 결과 표시
        label = QLabel(result_dialog)
        label.setPixmap(pixmap.scaled(dialog_width, dialog_height, Qt.AspectRatioMode.KeepAspectRatio))
        
        # 좌표 결과 표시 및 유사도 점수 추가
        result_text = "인식된 좌표:\n"
        for label_name, coords in center_points.items():
            result_text += f"{label_name}: ({coords[0]}, {coords[1]})\n"
        
        # 유사도 점수 추가
        if similarity is not None:
            # 백분율로 변환하여 표시 (0.0~1.0 → 0~100%)
            similarity_percent = similarity * 100
            result_text += f"\n템플릿 매칭 유사도: {similarity_percent:.2f}%"
        
        coord_label = QLabel(result_text, result_dialog)
        
        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(coord_label)
        result_dialog.setLayout(layout)
        
        # 결과 다이얼로그가 닫힐 때 부모 다이얼로그도 닫기
        result_dialog.finished.connect(lambda: self.parent_dialog.finish_process_and_close())
        
        result_dialog.exec()
    
    # 매칭 실패 처리를 위한 새 메서드 추가
    def show_matching_failed(self, cv2_img, similarity):
        """매칭 실패 메시지 표시"""
        # 화면 표시를 위해 RGB로 변환
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        height, width = img_rgb.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 대화 상자 생성
        failed_dialog = QDialog()
        failed_dialog.setWindowTitle("템플릿 매칭 실패")
        
        # 화면 크기의 절반으로 조정
        screen = QApplication.primaryScreen().geometry()
        dialog_width = screen.width() // 2
        dialog_height = screen.height() // 2
        failed_dialog.resize(dialog_width, dialog_height)
        
        # 이미지 표시
        label = QLabel(failed_dialog)
        label.setPixmap(pixmap.scaled(dialog_width, dialog_height, Qt.AspectRatioMode.KeepAspectRatio))
        
        # 실패 메시지 (유사도 포함)
        similarity_percent = similarity * 100
        message = f"화면에서 템플릿 이미지를 찾지 못했습니다.\n검출된 최대 유사도: {similarity_percent:.2f}%\n(필요 유사도: {self.TEMPLATE_MATCHING_THRESHOLD * 100:.0f}% 이상)"
        message_label = QLabel(message, failed_dialog)
        
        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(message_label)
        failed_dialog.setLayout(layout)
        
        # 결과 다이얼로그가 닫힐 때 부모 다이얼로그도 닫기
        failed_dialog.finished.connect(lambda: self.parent_dialog.finish_process_and_close())
        
        failed_dialog.exec()