import os
import threading
import time

import cv2
import mss
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from core_functions.mouse_controller import MouseController
from core_functions.screen_monitor import ScreenMonitor
from core_functions.vision_engine import VisionEngine
from utils.data_manager import DataManager
from utils.path_manager import debug_dir, resolve_project_path


class MacroRunner(QObject):
    # (macro_key, status)
    status_changed = pyqtSignal(str, str)
    # GUI 로그창에 보낼 문자열
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running_macros = {}  # {macro_key: Thread}
        self.stop_flags = {}  # {macro_key: threading.Event}
        self.data_manager = DataManager.get_instance()
        self.image_matcher = VisionEngine(mode="ocr", threshold=0.7)
        self.default_run_options = {
            "max_retries": 10,
            "retry_interval_sec": 0.5,
            "template_timeout_sec": 10.0,
            "save_debug_every_n_failures": 3,
        }

    def _emit_log(self, message):
        self.log_message.emit(str(message))

    def _merge_run_options(self, macro_data, run_options):
        merged = self.default_run_options.copy()
        macro_settings = macro_data.get("settings", {})

        merged["max_retries"] = macro_settings.get("max_retries", merged["max_retries"])
        merged["retry_interval_sec"] = macro_settings.get(
            "retry_interval_sec",
            merged["retry_interval_sec"],
        )
        merged["template_timeout_sec"] = macro_settings.get(
            "template_timeout_sec",
            merged["template_timeout_sec"],
        )

        if run_options:
            merged.update(run_options)

        return merged

    def _apply_matcher_mode(self, macro_data):
        macro_settings = macro_data.get("settings", {})
        matcher_mode = macro_settings.get("matcher_mode", "ocr")
        self.image_matcher.set_mode(matcher_mode)
        self.image_matcher.reload_templates()

    def start_macro(self, macro_key, run_options=None):
        """매크로 1개를 워커 스레드에서 시작한다."""
        if macro_key in self.running_macros and self.running_macros[macro_key].is_alive():
            self._emit_log(f"이미 실행 중인 매크로입니다: {macro_key}")
            return False

        macro_data = self.data_manager._data["macro_list"].get(macro_key)
        if not macro_data:
            self._emit_log(f"매크로를 찾을 수 없습니다: {macro_key}")
            return False

        merged_options = self._merge_run_options(macro_data, run_options)
        self._apply_matcher_mode(macro_data)

        stop_event = threading.Event()
        self.stop_flags[macro_key] = stop_event

        thread = threading.Thread(
            target=self._run_macro,
            args=(macro_key, macro_data, stop_event, merged_options),
        )
        thread.daemon = True
        thread.start()

        self.running_macros[macro_key] = thread
        self.status_changed.emit(macro_key, "running")
        return True

    def stop_macro(self, macro_key):
        """매크로 1개의 중지를 요청한다."""
        stop_event = self.stop_flags.get(macro_key)
        if stop_event is None:
            return False

        stop_event.set()

        thread = self.running_macros.get(macro_key)
        if thread is not None:
            thread.join(1.0)
            if not thread.is_alive():
                self.running_macros.pop(macro_key, None)

        self.status_changed.emit(macro_key, "stopped")
        return True

    def stop_all(self):
        """실행 중인 모든 매크로의 중지를 요청한다."""
        for macro_key in list(self.stop_flags.keys()):
            self.stop_macro(macro_key)

    def _save_debug_images(self, template_id, location=None, confidence=None, is_success=False):
        """
        디버그 이미지를 저장한다.

        - 성공 시: 매칭된 템플릿 박스 + 액션 영역 표시
        - 실패 시: matchTemplate 기준 최고 점수 후보 표시
        """
        output_dir = debug_dir()

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            prefix = "success" if is_success else "failed"

            with mss.mss() as sct:
                monitor = sct.monitors[0]
                screenshot = np.array(sct.grab(monitor))
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            template_img = None
            if template_id in self.image_matcher.template_paths:
                template_path = self.image_matcher.template_paths[template_id]
                template_img = cv2.imread(template_path)

            if template_img is None:
                return

            if is_success and location:
                self._save_success_debug_image(
                    output_dir,
                    timestamp,
                    screenshot,
                    template_img,
                    template_id,
                    location,
                    confidence,
                )
                return

            self._save_fail_debug_image(output_dir, timestamp, screenshot, template_img, template_id)

        except Exception as debug_error:
            error_path = output_dir / "error_log.txt"
            with open(error_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"디버그 이미지 저장 오류: {debug_error}\n"
                )

    def _save_success_debug_image(
        self,
        output_dir,
        timestamp,
        screenshot,
        template_img,
        template_id,
        location,
        confidence,
    ):
        h, w = template_img.shape[:2]
        debug_img = screenshot.copy()

        # 템플릿 매칭 성공 영역 표시
        cv2.rectangle(debug_img, location, (location[0] + w, location[1] + h), (0, 255, 0), 2)

        center_x = location[0] + w // 2
        center_y = location[1] + h // 2
        cross_size = 10
        cv2.line(
            debug_img,
            (center_x - cross_size, center_y),
            (center_x + cross_size, center_y),
            (0, 255, 0),
            2,
        )
        cv2.line(
            debug_img,
            (center_x, center_y - cross_size),
            (center_x, center_y + cross_size),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            debug_img,
            "템플릿",
            (location[0], location[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        score_text = f"점수: {confidence:.3f}" if confidence is not None else "점수: n/a"
        cv2.putText(debug_img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if template_id in self.image_matcher.template_actions:
            scale_info = None
            if template_id in self.image_matcher.template_sizes:
                orig_width, orig_height = self.image_matcher.template_sizes[template_id]
                scale_x = w / orig_width if orig_width > 0 else 1.0
                scale_y = h / orig_height if orig_height > 0 else 1.0
                scale_info = (scale_x, scale_y, w, h)

            for action_id in self.image_matcher.template_actions[template_id]:
                coords = self.image_matcher.get_scaled_action_coordinates(
                    template_id,
                    action_id,
                    location,
                    scale_info,
                )
                if not coords:
                    continue

                x, y, width, height = coords
                # 기존 코드에서 쓰던 x2 표시 방식을 유지 (증빙 이미지 비교용)
                cv2.rectangle(
                    debug_img,
                    (x * 2, y * 2),
                    (x * 2 + width * 2, y * 2 + height * 2),
                    (255, 0, 0),
                    2,
                )

                center = self.image_matcher.get_action_center(
                    template_id,
                    action_id,
                    location,
                    scale_info,
                )
                if center:
                    action_center_x, action_center_y = center
                    cv2.line(
                        debug_img,
                        (action_center_x - 5, action_center_y),
                        (action_center_x + 5, action_center_y),
                        (255, 0, 0),
                        2,
                    )
                    cv2.line(
                        debug_img,
                        (action_center_x, action_center_y - 5),
                        (action_center_x, action_center_y + 5),
                        (255, 0, 0),
                        2,
                    )

                cv2.putText(
                    debug_img,
                    action_id,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    1,
                )

        cv2.imwrite(str(output_dir / f"success_{timestamp}.png"), debug_img)

    def _save_fail_debug_image(self, output_dir, timestamp, screenshot, template_img, template_id):
        result = cv2.matchTemplate(screenshot, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        debug_img = screenshot.copy()
        h, w = template_img.shape[:2]
        cv2.rectangle(debug_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)

        threshold = getattr(self.image_matcher.active_matcher, "threshold", None)
        if threshold is None:
            threshold = getattr(self.image_matcher, "threshold", 0.7)

        score_text = f"Score: {max_val:.3f} (< {threshold})"
        cv2.putText(debug_img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(str(output_dir / f"failed_{timestamp}.png"), debug_img)

    def _find_original_image_action(self, actions):
        """
        기준 템플릿(원본 이미지) 액션을 찾는다.

        예전 저장 데이터는 액션 이름 문자열이 다를 수 있어서
        표시 이름만 믿지 않고 number/type도 같이 본다.
        """
        if not isinstance(actions, dict):
            return None, None

        # 1) 이 프로젝트에서 가장 안정적인 기준: number == 1 인 image 액션
        for action_key, action_data in actions.items():
            if not isinstance(action_data, dict):
                continue
            if action_data.get("type") == "image" and action_data.get("number") == 1:
                return action_key, action_data

        # 2) fallback: 첫 번째 image 액션
        for action_key, action_data in actions.items():
            if isinstance(action_data, dict) and action_data.get("type") == "image":
                return action_key, action_data

        return None, None

    def _run_macro(self, macro_key, macro_data, stop_flag, run_options):
        """워커 스레드에서 실제 매크로를 실행하는 함수."""
        self._emit_log(f"매크로 시작: {macro_data.get('name', macro_key)}")

        screen_monitor = ScreenMonitor(self.image_matcher)
        mouse_controller = MouseController()

        try:
            actions = macro_data.get("actions", {})
            original_action_key, original_action = self._find_original_image_action(actions)

            if not original_action_key or not original_action:
                self._emit_log("이 매크로에서 기준 이미지 액션을 찾지 못했습니다.")
                return

            image_path = resolve_project_path(original_action.get("image"))
            if not image_path or not os.path.exists(image_path):
                self._emit_log(f"기준 이미지 파일이 없습니다: {original_action.get('image')}")
                return

            template_id = f"{macro_key}_{original_action_key}"

            retry_count = 0
            max_retries = int(run_options.get("max_retries", 10))
            retry_interval_sec = float(run_options.get("retry_interval_sec", 0.5))
            template_timeout_sec = float(run_options.get("template_timeout_sec", 10.0))
            save_debug_every_n_failures = max(
                1,
                int(run_options.get("save_debug_every_n_failures", 3)),
            )
            started_at = time.time()

            # ScreenMonitor 생성 자체가 초기화 의미가 있어서 유지 (현재는 직접 호출 안 함)
            _ = screen_monitor

            while not stop_flag.is_set() and retry_count < max_retries:
                elapsed = time.time() - started_at
                if elapsed >= template_timeout_sec:
                    self._emit_log(
                        f"템플릿 탐색 타임아웃 ({template_timeout_sec:.1f}초)"
                    )
                    break

                result = self.image_matcher.find_template(template_id)
                found = bool(result and len(result) > 0 and result[0])

                if found:
                    location = result[1]
                    confidence = result[2] if len(result) > 2 else None
                    scale_info = result[4] if len(result) > 4 else None

                    if confidence is None:
                        self._emit_log(f"Template found at ({location[0]}, {location[1]})")
                    else:
                        self._emit_log(
                            f"템플릿 발견: ({location[0]}, {location[1]}), "
                            f"점수={confidence:.2f}"
                        )

                    if scale_info and len(scale_info) >= 2:
                        scale_x, scale_y = scale_info[0], scale_info[1]
                        self._emit_log(
                            f"스케일 정보: {scale_x:.4f} x {scale_y:.4f}"
                        )

                    self._save_debug_images(
                        template_id,
                        location=location,
                        confidence=confidence,
                        is_success=True,
                    )

                    success_count, fail_count = mouse_controller.click_all_actions(
                        self.image_matcher,
                        template_id,
                        fixed_location=location,
                        fixed_scale_info=scale_info,
                    )
                    self._emit_log(
                        f"액션 클릭 결과 - 성공: {success_count}, 실패: {fail_count}"
                    )
                    break

                retry_count += 1

                if retry_count == 1:
                    self._emit_log("템플릿을 찾지 못했습니다. 재시도합니다...")

                # N번마다 실패 디버그 이미지 저장 (기존 패턴 유지)
                if retry_count % save_debug_every_n_failures == 1:
                    self._save_debug_images(template_id, is_success=False)

                time.sleep(retry_interval_sec)

                if retry_count >= max_retries:
                    self._emit_log(
                        f"최대 재시도 횟수({max_retries})에 도달하여 중지합니다."
                    )

        except Exception as e:
            import traceback

            self._emit_log(f"매크로 실행 중 오류: {e}")
            with open(debug_dir() / "error_log.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{traceback.format_exc()}\n"
                )
        finally:
            self._emit_log(f"매크로 종료: {macro_data.get('name', macro_key)}")
            self.stop_flags.pop(macro_key, None)
            self.running_macros.pop(macro_key, None)
            self.status_changed.emit(macro_key, "stopped")
