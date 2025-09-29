import time
import threading

class ScreenMonitor:
    def __init__(self, image_matcher, template=None, max_duration=10, interval=0.2, callback=None):
        """
        화면 주시 모듈 초기화
        
        Args:
            image_matcher: ImageMatcher 인스턴스
            template: 찾을 템플릿 (template_id 또는 이미지, None이면 나중에 설정)
            max_duration: 최대 모니터링 시간 (초)
            interval: 스캔 간격 (초)
            callback: 매칭 발견 시 호출할 콜백 함수 (매개변수: success, location, confidence, scale_info)
        """
        self.matcher = image_matcher
        self.template = template
        self.max_duration = max_duration
        self.interval = interval
        self.callback = callback
        self.monitor_thread = None
        self.stop_event = threading.Event()
    
    def start_monitoring(self, template=None, callback=None, max_duration=None, interval=None):
        """
        화면 모니터링 시작
        
        Args:
            template: 찾을 템플릿 (None이면 이전에 설정된 템플릿 사용)
            callback: 콜백 함수 (None이면 이전에 설정된 콜백 사용)
            max_duration: 최대 모니터링 시간 (None이면 이전 설정 사용)
            interval: 스캔 간격 (None이면 이전 설정 사용)
        """
        # 이전 모니터링 중지
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join()
            
        # 설정 업데이트
        if template is not None:
            self.template = template
        if callback is not None:
            self.callback = callback
        if max_duration is not None:
            self.max_duration = max_duration
        if interval is not None:
            self.interval = interval
            
        # 필수 값 확인
        if self.template is None:
            raise ValueError("템플릿이 설정되지 않았습니다.")
            
        # 이벤트 초기화
        self.stop_event.clear()
        
        # 모니터링 스레드 시작
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        return self.monitor_thread
    
    def stop_monitoring(self):
        """모니터링 중지"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            return True
        return False
    
    def _monitor_loop(self):
        """모니터링 루프 (내부 사용)"""
        start_time = time.time()
        
        while not self.stop_event.is_set():
            # 시간 초과 확인
            if time.time() - start_time > self.max_duration:
                if self.callback:
                    self.callback(False, None, 0.0, None)
                break
                
            # 이미지 매칭 시도
            success, location, confidence, _, scale_info = self.matcher.find_template(self.template)
            
            # 이미지 발견 시
            if success:
                if self.callback:
                    self.callback(True, location, confidence, scale_info)
                break
                
            # 잠시 대기
            time.sleep(self.interval)
    
    def wait_for_template_and_click_actions(self, template_id, mouse_controller, max_duration=None):
        """
        템플릿을 기다리고 관련 액션을 모두 클릭
        
        Args:
            template_id: 템플릿 ID
            mouse_controller: MouseController 인스턴스
            max_duration: 최대 대기 시간 (None이면 기본값 사용)
            
        Returns:
            (성공 여부, 클릭한 액션 수)
        """
        result = [False, 0]
        
        def on_template_found(success, location, confidence, scale_info):
            if success:
                result[0] = True
                success_count, _ = mouse_controller.click_all_actions(self.matcher, template_id)
                result[1] = success_count
        
        # 모니터링 시작
        self.start_monitoring(
            template=template_id, 
            callback=on_template_found,
            max_duration=max_duration
        )
        
        # 모니터링 완료 대기
        if self.monitor_thread:
            self.monitor_thread.join()
            
        return result