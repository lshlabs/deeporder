import cv2
import numpy as np
import mss
import json
import os
from PyQt6.QtGui import QImage, QPixmap
import time

class ImageMatcher:
    def __init__(self, threshold=0.7, data_file_path="deeporder/utils/data.json"):
        """
        이미지 매칭 모듈 초기화
        
        Args:
            threshold (float): 템플릿 매칭 임계값 (0.0~1.0)
            data_file_path (str): data.json 파일 경로
        """
        self.threshold = threshold
        self.data_file_path = data_file_path
        self.templates = {}  # 로드된 템플릿 이미지 저장 (key: template_id, value: cv2_image)
        self.template_paths = {}  # 템플릿 이미지 경로 저장
        self.template_sizes = {}  # 원본 템플릿 크기 저장 (data.json 기준)
        self.template_actions = {}  # 템플릿과 관련된 액션들 저장
        
        # data.json에서 템플릿 경로와 좌표 정보 로드
        self.load_template_data()
    
    def load_template_data(self):
        """data.json 파일에서 템플릿 이미지 경로와 관련 액션 정보 로드"""
        try:
            with open(self.data_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # 템플릿 경로와 액션 정보 추출
            for macro_key, macro_data in data.get('macro_list', {}).items():
                actions = macro_data.get('actions', {})
                
                # 원본 이미지 찾기
                original_template = None
                original_action_key = None
                
                for action_key, action_data in actions.items():
                    if isinstance(action_data, dict) and action_data.get('name') == "원본 이미지":
                        original_template = action_data
                        original_action_key = action_key
                        break
                
                if not original_template:
                    continue
                
                # 원본 이미지 경로 확인
                image_path = original_template.get('image')
                if not image_path or not os.path.exists(image_path):
                    continue
                
                # 템플릿 ID 생성 (매크로 키 + 액션 키)
                template_id = f"{macro_key}_{original_action_key}"
                
                # 템플릿 경로 저장
                self.template_paths[template_id] = image_path
                
                # 원본 이미지 크기 저장 (data.json의 coordinates 기준)
                coords = original_template.get('coordinates', [0, 0, 0, 0])
                if len(coords) >= 4:
                    w, h = coords[2], coords[3]
                    self.template_sizes[template_id] = (w, h)
                    print(f"템플릿 ID: {template_id}, 원본 크기: {w}x{h}")
                
                # 템플릿과 관련된 액션들 저장 (원본 이미지 제외)
                related_actions = {}
                for action_key, action_data in actions.items():
                    if isinstance(action_data, dict) and action_key != original_action_key:
                        related_actions[action_key] = action_data
                
                self.template_actions[template_id] = related_actions
        except Exception as e:
            print(f"템플릿 데이터 로드 실패: {e}")
    
    def load_template(self, template_id):
        """템플릿 ID에 해당하는 이미지 로드"""
        if template_id not in self.template_paths:
            return None
        
        if template_id in self.templates:
            return self.templates[template_id]
        
        try:
            path = self.template_paths[template_id]
            template = cv2.imread(path)
            self.templates[template_id] = template
            return template
        except Exception as e:
            print(f"템플릿 이미지 로드 실패: {e}")
            return None
    
    def capture_screen(self):
        """현재 화면 캡처"""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"화면 캡처 실패: {e}")
            return None
    
    def find_template(self, template_id):
        """
        화면에서 템플릿 이미지 찾기
        
        Args:
            template_id: 템플릿 ID
            
        Returns:
            (success, location, confidence, screenshot, scale_info)
        """
        # 템플릿 로드
        template = self.load_template(template_id)
        if template is None:
            return False, None, 0.0, None, None
        
        # 템플릿 크기 확인
        template_h, template_w = template.shape[:2]
        
        # 전체 화면 캡처
        screenshot = self.capture_screen()
        if screenshot is None:
            return False, None, 0.0, None, None
        
        # 스크린샷 크기
        screenshot_h, screenshot_w = screenshot.shape[:2]
        
        # 템플릿 매칭 수행
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 임계값 검사
        if max_val < self.threshold:
            return False, None, max_val, screenshot, None
        
        # 스케일 계산 (data.json의 원본 크기 vs 실제 템플릿 이미지 크기)
        scale_info = None
        if template_id in self.template_sizes:
            orig_width, orig_height = self.template_sizes[template_id]
            
            # X, Y 스케일 계산
            scale_x = template_w / orig_width if orig_width > 0 else 1.0
            scale_y = template_h / orig_height if orig_height > 0 else 1.0
            
            # 스케일 정보 저장
            scale_info = (scale_x, scale_y, template_w, template_h)
            
            # 디버깅 메시지
            print(f"[중요] 템플릿 실제/원본 스케일: {scale_x:.4f}x{scale_y:.4f}")
            print(f"[중요] 원본 크기: {orig_width}x{orig_height}, 실제 크기: {template_w}x{template_h}")
        
        return True, max_loc, max_val, screenshot, scale_info
    
    def get_scaled_action_coordinates(self, template_id, action_id, template_location, scale_info):
        """
        템플릿 내부 액션의 좌표 계산 (스케일 적용)
        
        Args:
            template_id: 템플릿 ID
            action_id: 액션 ID
            template_location: 화면에서 찾은 템플릿 위치 (x, y)
            scale_info: 스케일 정보 (scale_x, scale_y, template_w, template_h)
            
        Returns:
            (x, y, width, height): 스케일 적용된 좌표
        """
        if template_id not in self.template_actions:
            return None
        
        actions = self.template_actions[template_id]
        if action_id not in actions:
            return None
        
        action_data = actions[action_id]
        coordinates = action_data.get('coordinates')
        if not coordinates or len(coordinates) < 4:
            return None
        
        # 원본 좌표
        orig_x, orig_y, orig_width, orig_height = coordinates
        
        # 기본값
        scaled_x = orig_x
        scaled_y = orig_y
        scaled_width = orig_width
        scaled_height = orig_height
        
        # 스케일 적용
        if scale_info and len(scale_info) >= 2:
            scale_x, scale_y = scale_info[0], scale_info[1]
            
            # 좌표에 스케일 적용
            scaled_x = int(orig_x * scale_x)
            scaled_y = int(orig_y * scale_y)
            scaled_width = int(orig_width * scale_x)
            scaled_height = int(orig_height * scale_y)
        
        # 템플릿 위치 기준으로 절대 좌표 계산 (위치를 2로 나눔)
        abs_x = template_location[0] // 2 + scaled_x
        abs_y = template_location[1] // 2 + scaled_y
        
        # 좌표 정보 출력
        print(f"액션 '{action_id}' 좌표: 원본=({orig_x},{orig_y},{orig_width},{orig_height}), "
              f"스케일 적용=({scaled_x},{scaled_y},{scaled_width},{scaled_height}), "
              f"절대 좌표=({abs_x},{abs_y},{scaled_width},{scaled_height})")
        
        return (abs_x, abs_y, scaled_width, scaled_height)
    
    def get_action_center(self, template_id, action_id, template_location, scale_info):
        """
        템플릿 내부 액션의 중심점 계산
        
        Args:
            template_id: 템플릿 ID
            action_id: 액션 ID
            template_location: 화면에서 찾은 템플릿 위치 (x, y)
            scale_info: 스케일 정보
            
        Returns:
            (center_x, center_y): 중심점 좌표
        """
        coords = self.get_scaled_action_coordinates(template_id, action_id, template_location, scale_info)
        if coords is None:
            return None
        
        x, y, width, height = coords
        center_x = x + width // 2
        center_y = y + height // 2
        
        return (center_x, center_y)
    
    def get_all_action_centers(self, template_id, template_location, scale_info):
        """
        템플릿과 관련된 모든 액션의 중심점 계산
        
        Args:
            template_id: 템플릿 ID
            template_location: 화면에서 찾은 템플릿 위치 (x, y)
            scale_info: 스케일 정보
            
        Returns:
            {action_id: (center_x, center_y), ...}
        """
        if template_id not in self.template_actions:
            return {}
        
        centers = {}
        for action_id in self.template_actions[template_id]:
            center = self.get_action_center(template_id, action_id, template_location, scale_info)
            if center:
                centers[action_id] = center
        
        return centers