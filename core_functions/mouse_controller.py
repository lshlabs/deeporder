import time
import pyautogui

class MouseController:
    def __init__(self, click_delay=0.1):
        """
        마우스 컨트롤러 초기화
        
        Args:
            click_delay: 클릭 전후 딜레이 (초)
        """
        self.click_delay = click_delay
        pyautogui.PAUSE = click_delay
    
    def click_at(self, x, y, button='left', clicks=1, delay=None):
        """
        지정된 좌표에서 마우스 클릭
        
        Args:
            x, y: 클릭할 화면 좌표
            button: 'left', 'right', 'middle' 중 하나
            clicks: 클릭 횟수
            delay: 클릭간 딜레이 (None이면 기본값 사용)
        """
        if delay is None:
            delay = self.click_delay
            
        # 마우스 이동
        pyautogui.moveTo(x, y)
        time.sleep(delay)
        
        # 클릭
        pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=delay)
        time.sleep(delay)
        
        return True
    
    def click_at_template(self, matcher, template_id, button='left', clicks=1, offset=(0, 0)):
        """
        템플릿 이미지 위치를 찾아 클릭
        
        Args:
            matcher: ImageMatcher 인스턴스
            template_id: 찾을 템플릿 ID
            button: 마우스 버튼
            clicks: 클릭 횟수
            offset: 템플릿 중심에서의 오프셋 (dx, dy)
        
        Returns:
            성공 여부
        """
        # 템플릿 찾기
        success, location, _, _, scale_info = matcher.find_template(template_id)
        
        if not success:
            return False
            
        # 템플릿 이미지 로드 및 중심점 계산
        template = matcher.load_template(template_id)
        if template is None:
            return False
            
        h, w = template.shape[:2]
        center_x = location[0] + w // 2
        center_y = location[1] + h // 2
            
        # 오프셋 적용
        center_x += offset[0]
        center_y += offset[1]
        
        # 클릭
        return self.click_at(center_x, center_y, button, clicks)
    
    def click_at_action(self, matcher, template_id, action_id, button='left', clicks=1, offset=(0, 0)):
        """
        템플릿 내 특정 액션 위치를 찾아 클릭
        
        Args:
            matcher: ImageMatcher 인스턴스
            template_id: 템플릿 ID
            action_id: 액션 ID
            button: 마우스 버튼
            clicks: 클릭 횟수
            offset: 액션 중심에서의 오프셋 (dx, dy)
        
        Returns:
            성공 여부
        """
        # 템플릿 찾기
        success, location, _, _, scale_info = matcher.find_template(template_id)
        
        if not success:
            return False
        
        # 액션 중심점 계산
        center = matcher.get_action_center(template_id, action_id, location, scale_info)
        if center is None:
            return False
            
        # 오프셋 적용
        center_x = center[0] + offset[0]
        center_y = center[1] + offset[1]
        
        # 클릭
        return self.click_at(center_x, center_y, button, clicks)
    
    def click_all_actions(self, matcher, template_id, fixed_location=None, fixed_scale_info=None):
        """
        템플릿 내 모든 액션 클릭
        
        Args:
            matcher: ImageMatcher 인스턴스
            template_id: 템플릿 ID
            fixed_location: 이미 알고 있는 템플릿 위치 (다시 찾지 않음)
            fixed_scale_info: 이미 알고 있는 스케일 정보 (다시 계산하지 않음)
        """
        if fixed_location is None or fixed_scale_info is None:
            # 템플릿 찾기
            success, location, _, _, scale_info = matcher.find_template(template_id)
            if not success:
                return 0, 0
        else:
            # 제공된 위치와 스케일 사용
            location = fixed_location
            scale_info = fixed_scale_info
        
        # 템플릿 액션 정보 확인
        if template_id not in matcher.template_actions:
            return 0, 0
            
        # 액션 순서대로 클릭
        success_count = 0
        fail_count = 0
        
        # number 순서로 정렬
        sorted_actions = sorted(
            matcher.template_actions[template_id].items(),
            key=lambda x: x[1].get('number', 0)
        )
        
        for action_id, action_data in sorted_actions:
            # 액션 중심점 계산
            center = matcher.get_action_center(template_id, action_id, location, scale_info)
            if center is None:
                fail_count += 1
                continue
            
            # 클릭
            if self.click_at(center[0], center[1]):
                success_count += 1
            else:
                fail_count += 1
            
        return success_count, fail_count