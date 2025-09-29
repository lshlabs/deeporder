import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import cv2
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt6.QtWidgets import QLabel, QDialog
import mss
import time

from example.testcode import WizardStep2Dialog

class WizardStep2DialogExtended(WizardStep2Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.countdown = 5
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_countdown)
        self.countdown_dialog = None
        
    def save_and_close(self):
        # 부모 클래스의 save_and_close 실행
        super().save_and_close()
        
        # 카운트다운 다이얼로그 생성 및 표시
        self.countdown = 5
        self.countdown_dialog = QtWidgets.QDialog(self)
        self.countdown_dialog.setWindowTitle("스크린샷 준비")
        self.countdown_label = QLabel(f"{self.countdown}초 후에 스크린샷이 촬영됩니다.", self.countdown_dialog)
        self.countdown_dialog.setLayout(QtWidgets.QVBoxLayout())
        self.countdown_dialog.layout().addWidget(self.countdown_label)
        self.countdown_dialog.show()
        
        # 타이머 시작
        self.timer.start(1000)

    def update_countdown(self):
        self.countdown -= 1
        self.countdown_label.setText(f"{self.countdown}초 후에 스크린샷이 촬영됩니다.")
        
        if self.countdown <= 0:
            self.timer.stop()
            self.countdown_dialog.close()
            self.take_screenshot()

    def take_screenshot(self):
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
            
            # 매칭된 위치 (이것이 기준점이 됨)
            template_x, template_y = max_loc
            
            # 스케일 비율 계산 (스케일된 크기 -> 원본 크기로 변환하기 위한 비율)
            scale_x = self.original_pixmap.width() / self.display_pixmap.width()
            scale_y = self.original_pixmap.height() / self.display_pixmap.height()
            
            print("\n=== 각 영역의 중심 좌표 ===")
            # 각 영역을 빨간색으로 표시
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
                    
                    print(f"{label} 영역 중심점: ({center_x}, {center_y})")
                    
                    # 빨간색 사각형 그리기
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # 중심점에 빨간색 점 그리기 (반지름 5픽셀의 원)
                    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 템플릿 영역 전체를 표시
            h, w = template.shape[:2]
            cv2.rectangle(img, (template_x, template_y), 
                        (template_x + w, template_y + h), (0, 0, 255), 2)
            
            # 원본 크기로 이미지 저장 (BGR 형식)
            cv2.imwrite('screenshot_result.png', img)
            
            # 화면 표시를 위해 RGB로 변환
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 결과 이미지를 QPixmap으로 변환하여 새 창에 표시
            self.show_result(img_rgb)

    def convert_qpixmap_to_cv2(self, pixmap):
        # QPixmap을 CV2 이미지로 변환
        qimage = pixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def show_result(self, cv2_img):
        height, width = cv2_img.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(cv2_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 새 창 생성 및 이미지 표시
        result_dialog = QDialog()
        result_dialog.setWindowTitle("스크린샷 결과")
        
        # 화면 크기의 절반으로 조정
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        dialog_width = screen.width() // 2
        dialog_height = screen.height() // 2
        result_dialog.resize(dialog_width, dialog_height)
        
        # 레이블에 이미지 표시
        label = QLabel(result_dialog)
        label.setPixmap(pixmap.scaled(dialog_width, dialog_height, Qt.AspectRatioMode.KeepAspectRatio))
        
        # 레이아웃 설정
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        result_dialog.setLayout(layout)
        
        result_dialog.exec()

def main():
    app = QtWidgets.QApplication(sys.argv)
    dialog = WizardStep2DialogExtended()
    
    # 테스트를 위한 이미지 설정
    test_image = QPixmap('/Users/mac/Documents/GitHub/myGit/deeporder/example/쿠팡이츠 배달 레티나.png')
    dialog.original_pixmap = test_image
    dialog.display_pixmap = test_image.copy()
    dialog.label_preview.setPixmap(dialog.display_pixmap)
    dialog.update_preview()
    
    dialog.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
