import threading
from deeporder.core_functions.screen_monitor import ScreenMonitor
from deeporder.core_functions.mouse_controller import MouseController

class MacroScheduler:
    def __init__(self, matcher):
        self.matcher = matcher
        self.monitors = {}
        self.active_macro = None
        self.pending_macros = []  # 대기열 추가
        self.action_lock = threading.Lock()
    
    def add_macro(self, macro_id, max_duration=30):
        """매크로 모니터링 시작"""
        # 각 매크로별 전용 모니터 생성
        self.monitors[macro_id] = ScreenMonitor(self.matcher, max_duration=max_duration)
        
        # 해당 매크로의 콜백 설정
        def on_template_found(success, location, confidence, scale_info):
            if not success:
                return
                
            with self.action_lock:
                # 현재 실행 중인 매크로가 있는지 확인
                if self.active_macro is not None and self.active_macro != macro_id:
                    print(f"매크로 {macro_id} 발견됨, 하지만 {self.active_macro}가 실행 중. 대기열에 추가.")
                    
                    # 대기열에 없는 경우만 추가 (중복 방지)
                    if macro_id not in self.pending_macros:
                        self.pending_macros.append(macro_id)
                    return
                
                # 액션 실행 시작
                self._execute_macro(macro_id, location, scale_info)
        
        # 모니터링 시작 (템플릿 ID 설정)
        template_id = f"{macro_id}_A1"
        self.monitors[macro_id].start_monitoring(
            template=template_id, 
            callback=on_template_found,
            max_duration=None  # 무제한 모니터링
        )
        print(f"매크로 {macro_id} 모니터링 시작")
    
    def _execute_macro(self, macro_id, location, scale_info):
        """매크로 액션 실행 (내부 사용)"""
        self.active_macro = macro_id
        print(f"매크로 {macro_id} 실행 시작")
        
        mouse = MouseController()
        template_id = f"{macro_id}_A1"
        
        # 모든 액션 클릭
        success_count, fail_count = mouse.click_all_actions(
            self.matcher, template_id, 
            fixed_location=location, fixed_scale_info=scale_info
        )
        
        print(f"매크로 {macro_id} 완료: 성공={success_count}, 실패={fail_count}")
        
        # 액션 완료 후 대기열 처리
        self.active_macro = None
        
        # 대기 중인 매크로가 있으면 다음 실행
        if self.pending_macros:
            next_macro = self.pending_macros.pop(0)
            print(f"대기열에서 다음 매크로 {next_macro} 실행")
            
            # 템플릿 다시 검색 (위치가 변경되었을 수 있음)
            success, location, _, _, scale_info = self.matcher.find_template(f"{next_macro}_A1")
            if success:
                self._execute_macro(next_macro, location, scale_info)
            else:
                print(f"대기 중이던 매크로 {next_macro}의 템플릿을 찾을 수 없음")