import threading
from PyQt6.QtCore import QObject, pyqtSignal
from core_functions.screen_monitor import ScreenMonitor
from core_functions.mouse_controller import MouseController
import os
from utils.data_manager import DataManager
import time
import mss
import numpy as np
import cv2
from core_functions.vision_engine import VisionEngine
from utils.path_manager import debug_dir, resolve_project_path

class MacroRunner(QObject):
    # 시그널 정의
    status_changed = pyqtSignal(str, str)  # (macro_key, status)
    log_message = pyqtSignal(str)  # 로그 메시지

    def __init__(self):
        super().__init__()
        self.running_macros = {}  # {macro_key: thread}
        self.stop_flags = {}      # {macro_key: stop_flag}
        self.data_manager = DataManager.get_instance()
        self.image_matcher = VisionEngine(mode="ocr", threshold=0.7)
        self.default_run_options = {
            "max_retries": 10,
            "retry_interval_sec": 0.5,
            "template_timeout_sec": 10.0,
            "save_debug_every_n_failures": 3,
        }
        
    def _emit_log(self, message):
        self.log_message.emit(message)
        
    def start_macro(self, macro_key, run_options=None):
        """매크로 실행 시작"""
        if macro_key in self.running_macros and self.running_macros[macro_key].is_alive():
            self._emit_log(f"이미 실행 중인 매크로입니다: {macro_key}")
            return False
            
        # 매크로 데이터 가져오기
        macro_data = self.data_manager._data['macro_list'].get(macro_key)
        if not macro_data:
            self._emit_log(f"매크로를 찾을 수 없습니다: {macro_key}")
            return False

        merged_options = self.default_run_options.copy()
        macro_settings = macro_data.get("settings", {})
        merged_options.update({
            "max_retries": macro_settings.get("max_retries", merged_options["max_retries"]),
            "retry_interval_sec": macro_settings.get("retry_interval_sec", merged_options["retry_interval_sec"]),
            "template_timeout_sec": macro_settings.get("template_timeout_sec", merged_options["template_timeout_sec"]),
        })
        if run_options:
            merged_options.update(run_options)
        matcher_mode = macro_settings.get("matcher_mode", "ocr")
        self.image_matcher.mode = matcher_mode if matcher_mode in {"ocr", "template"} else "ocr"
        self.image_matcher.reload_templates()
            
        # 중지 플래그 생성
        self.stop_flags[macro_key] = threading.Event()
        
        # 매크로 실행 스레드 생성 및 시작
        thread = threading.Thread(
            target=self._run_macro,
            args=(macro_key, macro_data, self.stop_flags[macro_key], merged_options)
        )
        thread.daemon = True
        thread.start()
        
        self.running_macros[macro_key] = thread
        self.status_changed.emit(macro_key, "running")
        return True
        
    def stop_macro(self, macro_key):
        """매크로 실행 중지"""
        if macro_key in self.stop_flags:
            self.stop_flags[macro_key].set()  # 중지 플래그 설정
            if macro_key in self.running_macros:
                self.running_macros[macro_key].join(1.0)  # 1초 대기
                if not self.running_macros[macro_key].is_alive():
                    del self.running_macros[macro_key]
                    
            self.status_changed.emit(macro_key, "stopped")
            return True
        return False

    def stop_all(self):
        for macro_key in list(self.stop_flags.keys()):
            self.stop_macro(macro_key)
        
    def _save_debug_images(self, template_id, location=None, confidence=None, is_success=False):
        """디버깅 이미지 저장"""
        debug_path = debug_dir()
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            prefix = "success" if is_success else "failed"
            
            # 스크린샷 캡처
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                screenshot = np.array(sct.grab(monitor))
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # 템플릿 이미지 로드
            template_img = None
            if template_id in self.image_matcher.template_paths:
                template_path = self.image_matcher.template_paths[template_id]
                template_img = cv2.imread(template_path)
            
            if template_img is None:
                return
                
            # 이미지 저장 (필요한 것만)
            if is_success and location:
                # 결과 이미지만 저장
                h, w = template_img.shape[:2]
                debug_img = screenshot.copy()
                
                # 성공: 찾은 위치에 초록색 사각형 (원본 이미지)
                cv2.rectangle(debug_img, location, (location[0] + w, location[1] + h), (0, 255, 0), 2)
                
                # 중심점 계산 및 표시
                center_x = location[0] + w // 2
                center_y = location[1] + h // 2
                
                # 중심점에 십자가 표시
                cross_size = 10
                cv2.line(debug_img, 
                        (center_x - cross_size, center_y),
                        (center_x + cross_size, center_y),
                        (0, 255, 0), 2)
                cv2.line(debug_img, 
                        (center_x, center_y - cross_size),
                        (center_x, center_y + cross_size),
                        (0, 255, 0), 2)
                
                # 원본 이미지 레이블 표시
                cv2.putText(debug_img, "원본 이미지", (location[0], location[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 신뢰도 점수 표시
                score_text = f"Score: {confidence:.3f}"
                cv2.putText(debug_img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 모든 액션 영역 표시 (파란색)
                if template_id in self.image_matcher.template_actions:
                    # 스케일 정보 계산
                    scale_info = None
                    if template_id in self.image_matcher.template_sizes:
                        orig_width, orig_height = self.image_matcher.template_sizes[template_id]
                        scale_x = w / orig_width if orig_width > 0 else 1.0
                        scale_y = h / orig_height if orig_height > 0 else 1.0
                        scale_info = (scale_x, scale_y, w, h)
                    
                    # 각 액션별로 MouseController가 사용하는 동일한 방식으로 좌표 계산
                    for action_id in self.image_matcher.template_actions[template_id]:
                        # ImageMatcher의 메서드 사용하여 액션 좌표 계산
                        coords = self.image_matcher.get_scaled_action_coordinates(
                            template_id, action_id, location, scale_info
                        )
                        
                        if coords:
                            x, y, width, height = coords
                            
                            # 액션 영역 사각형 그리기 (파란색)
                            cv2.rectangle(debug_img, (x*2, y*2), (x*2 + width*2, y*2 + height*2), (255, 0, 0), 2)
                            
                            # 액션 중심점 계산
                            center = self.image_matcher.get_action_center(
                                template_id, action_id, location, scale_info
                            )
                            
                            if center:
                                action_center_x, action_center_y = center
                                
                                # 중심점에 십자가 표시
                                cv2.line(debug_img, 
                                        (action_center_x - 5, action_center_y),
                                        (action_center_x + 5, action_center_y),
                                        (255, 0, 0), 2)
                                cv2.line(debug_img, 
                                        (action_center_x, action_center_y - 5),
                                        (action_center_x, action_center_y + 5),
                                        (255, 0, 0), 2)
                            
                            # 액션 ID 표시
                            cv2.putText(debug_img, action_id, (x, y - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                
                cv2.imwrite(str(debug_path / f"{prefix}_{timestamp}.png"), debug_img)
                
            elif is_success == False and template_img is not None:
                # 실패: 가장 유사한 위치에 빨간색 사각형 (매 3회마다만 저장)
                result = cv2.matchTemplate(screenshot, template_img, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                debug_img = screenshot.copy()
                h, w = template_img.shape[:2]
                cv2.rectangle(debug_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
                
                score_text = f"Score: {max_val:.3f} (< {self.image_matcher.threshold})"
                cv2.putText(debug_img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(str(debug_path / f"{prefix}_{timestamp}.png"), debug_img)
            
        except Exception as debug_e:
            # 디버깅 이미지 저장 실패는 주 로그에 출력하지 않음
            with open(debug_path / "error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 디버깅 이미지 저장 오류: {str(debug_e)}\n")

    def _run_macro(self, macro_key, macro_data, stop_flag, run_options):
        """매크로 실행 로직 (스레드에서 실행)"""
        self._emit_log(f"매크로 시작: {macro_data['name']}")
        
        screen_monitor = ScreenMonitor(self.image_matcher)
        mouse_controller = MouseController()
        
        try:
            # 원본 이미지(A1) 찾기
            original_image = None
            original_action_key = None
            actions = macro_data.get('actions', {})
            
            # 먼저 원본 이미지 액션 찾기
            for action_key, action_data in actions.items():
                if isinstance(action_data, dict) and action_data.get('name') == "원본 이미지":
                    original_image = action_data
                    original_action_key = action_key
                    break
            
            if not original_image:
                self._emit_log("원본 이미지를 찾을 수 없습니다.")
                return
            
            # 원본 이미지 경로 확인
            image_path = resolve_project_path(original_image.get('image'))
            if not image_path or not os.path.exists(image_path):
                self._emit_log(f"원본 이미지 파일이 존재하지 않습니다: {original_image.get('image')}")
                return
            
            # 템플릿 ID 생성
            template_id = f"{macro_key}_{original_action_key}"
            
            retry_count = 0
            max_retries = int(run_options.get("max_retries", 10))
            retry_interval_sec = float(run_options.get("retry_interval_sec", 0.5))
            template_timeout_sec = float(run_options.get("template_timeout_sec", 10.0))
            save_debug_every_n_failures = max(1, int(run_options.get("save_debug_every_n_failures", 3)))
            started_at = time.time()
            
            while not stop_flag.is_set() and retry_count < max_retries:
                if time.time() - started_at >= template_timeout_sec:
                    self._emit_log(f"탐색 타임아웃: {template_timeout_sec:.1f}초 내 원본 이미지를 찾지 못했습니다.")
                    break

                # 원본 이미지 찾기
                result = self.image_matcher.find_template(template_id)
                
                if result[0]:  # 성공
                    location = result[1]
                    confidence = result[2]
                    scale_info = result[4]
                    self._emit_log(f"원본 이미지 발견: 위치({location[0]}, {location[1]}), 신뢰도: {confidence:.2f}")
                    
                    # 스케일 정보 출력
                    if scale_info and len(scale_info) >= 2:
                        scale_x, scale_y = scale_info[0], scale_info[1]
                        self._emit_log(f"[중요] 템플릿 실제/원본 스케일: {scale_x:.4f}x{scale_y:.4f}")
                    
                    # 디버깅 이미지 저장 (성공)
                    self._save_debug_images(template_id, location, confidence, True)
                    
                    # 모든 액션 클릭 수행
                    success_count, fail_count = mouse_controller.click_all_actions(
                        self.image_matcher,
                        template_id,
                        fixed_location=location,
                        fixed_scale_info=scale_info
                    )
                    
                    self._emit_log(f"액션 클릭 결과: 성공 {success_count}, 실패 {fail_count}")
                    break
                else:
                    retry_count += 1
                    if retry_count == 1:  # 첫 실패 시에만 로그 메시지 표시
                        self._emit_log("원본 이미지를 찾지 못했습니다. 재시도 중...")
                    
                    # 디버깅 이미지 저장 (실패)
                    if retry_count % save_debug_every_n_failures == 1:
                        self._save_debug_images(template_id, is_success=False)
                    
                    time.sleep(retry_interval_sec)
                
                # 재시도 횟수 초과 시 메시지 출력
                if retry_count >= max_retries:
                    self._emit_log(f"최대 재시도 횟수({max_retries}회)를 초과했습니다. 매크로를 중단합니다.")
        
        except Exception as e:
            import traceback
            self._emit_log(f"매크로 실행 중 오류 발생: {str(e)}")
            # 상세 오류는 디버그 로그 파일에만 기록
            with open(debug_dir() / "error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 오류: {traceback.format_exc()}\n")
        finally:
            self._emit_log(f"매크로 종료: {macro_data['name']}")
            self.stop_flags.pop(macro_key, None)
            self.running_macros.pop(macro_key, None)
            self.status_changed.emit(macro_key, "stopped") 
