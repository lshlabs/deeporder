import cv2
import numpy as np
import pyautogui
import time
from datetime import datetime
import os
import Quartz  # macOS 창 관리를 위한 라이브러리

# 상수 정의
TARGET_PROGRAM = "Safari"  # 모니터링할 프로그램 이름
# 현재 파일의 절대 경로를 기준으로 이미지 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_IMAGE_PATH = os.path.join(SCRIPT_DIR, "..", "resources", "img", "chrome icon.png")
SEARCH_INTERVAL = 1.0  # 검색 주기 (초)

class WindowMonitor:
    def __init__(self):
        # 디버그 이미지를 저장할 디렉토리 생성
        self.debug_dir = os.path.join(SCRIPT_DIR, "debug_images")
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
            
    def get_safari_window_bounds(self):
        """Safari 창의 위치와 크기를 가져오는 함수"""
        # 모든 윈도우 정보 가져오기
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll,  # 모든 윈도우 가져오기
            Quartz.kCGNullWindowID
        )
        
        # Safari 창 찾기
        safari_windows = []
        for window in windows:
            owner_name = window.get(Quartz.kCGWindowOwnerName, '')
            window_layer = window.get(Quartz.kCGWindowLayer, 0)
            
            if owner_name == TARGET_PROGRAM:
                # 일반적인 앱 창의 레이어는 0
                if window_layer == 0:
                    bounds = window.get(Quartz.kCGWindowBounds)
                    safari_windows.append({
                        'x': int(bounds['X']),
                        'y': int(bounds['Y']),
                        'width': int(bounds['Width']),
                        'height': int(bounds['Height']),
                        'layer': window_layer
                    })
        
        # Safari 창이 없는 경우
        if not safari_windows:
            return None
        
        # 여러 Safari 창이 있는 경우 가장 큰 창을 선택 (메인 창일 가능성이 높음)
        main_window = max(safari_windows, 
                         key=lambda w: w['width'] * w['height'])
        
        return main_window

    def capture_safari_window(self):
        """Safari 창만 캡처하는 함수"""
        bounds = self.get_safari_window_bounds()
        if bounds is None:
            print(f"[{TARGET_PROGRAM}] 창을 찾을 수 없습니다.")
            return None
        
        try:
            # Safari 창 영역만 캡처
            screenshot = pyautogui.screenshot(region=(
                bounds['x'], bounds['y'], 
                bounds['width'], bounds['height']
            ))
            
            # PIL Image를 numpy array로 변환
            screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # 캡처된 이미지의 크기가 지정한 영역과 일치하는지 확인
            if screen.shape[:2] != (bounds['height'], bounds['width']):
                print("Error: 캡처된 이미지 크기가 Safari 창 크기와 다릅니다!")
                print(f"예상 크기: ({bounds['height']}, {bounds['width']})")
                print(f"실제 크기: {screen.shape[:2]}")
                return None
            
            # 디버깅을 위한 정보 출력
            print(f"캡처 영역: x={bounds['x']}, y={bounds['y']}, "
                  f"width={bounds['width']}, height={bounds['height']}")
            print(f"캡처된 이미지 크기: {screen.shape}")
            
            return screen
        
        except Exception as e:
            print(f"스크린샷 캡처 중 오류 발생: {e}")
            return None

    def find_target(self, target_image_path, threshold=0.8):
        """Safari 창 내에서 대상 이미지를 찾는 함수"""
        if not os.path.exists(target_image_path):
            raise ValueError(f"타겟 이미지 파일이 존재하지 않습니다: {target_image_path}")
        
        target = cv2.imread(target_image_path)
        if target is None:
            raise ValueError(f"타겟 이미지를 로드할 수 없습니다: {target_image_path}")
        
        # Safari 창만 캡처
        screen = self.capture_safari_window()
        if screen is None:
            return None
        
        # 디버깅을 위한 정보 출력
        print(f"화면 크기: {screen.shape}")
        print(f"타겟 이미지 크기: {target.shape}")
        
        # 템플릿 매칭 수행
        result = cv2.matchTemplate(screen, target, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"매칭 점수: {max_val}")  # 디버깅을 위한 매칭 점수 출력
        
        # 디버그 이미지 저장
        self.save_debug_image(screen, target, max_loc, max_val >= threshold)
        
        if max_val >= threshold:
            bounds = self.get_safari_window_bounds()
            target_h, target_w = target.shape[:2]
            # Safari 창 내부의 상대 좌표를 전체 화면의 절대 좌표로 변환
            center_x = bounds['x'] + max_loc[0] + target_w // 2
            center_y = bounds['y'] + max_loc[1] + target_h // 2
            return (center_x, center_y)
        
        return None
    
    def save_debug_image(self, screen, target, location, found):
        """디버그 이미지 저장"""
        debug_image = screen.copy()
        target_h, target_w = target.shape[:2]
        
        # 찾은 영역 표시
        color = (0, 255, 0) if found else (0, 0, 255)  # 찾음: 녹색, 못찾음: 빨간색
        cv2.rectangle(debug_image, location, 
                     (location[0] + target_w, location[1] + target_h), 
                     color, 2)
        
        # 현재 시간을 파일명으로 사용
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "found" if found else "not_found"
        filename = f"debug_{timestamp}_{status}.png"
        
        # 디버그 이미지 저장
        cv2.imwrite(os.path.join(self.debug_dir, filename), debug_image)

def main():
    monitor = WindowMonitor()
    
    try:
        while True:
            location = monitor.find_target(TARGET_IMAGE_PATH)
            if location:
                print(f"[{TARGET_PROGRAM}] 타겟 발견: {location}")
            else:
                print(f"[{TARGET_PROGRAM}] 타겟을 찾을 수 없습니다.")
            time.sleep(SEARCH_INTERVAL)
            
    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
