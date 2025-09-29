#!/usr/bin/env python3
"""
🎯 ROI 제한 시간 조절 버튼 감지
"거부"와 "접수" 버튼 사이 영역으로만 검색 범위를 제한하는 똑똑한 방법
"""

import cv2
import numpy as np
import easyocr
import mss
import time
from pathlib import Path
from typing import Tuple, Optional, Dict

class ROILimitedTimeController:
    """거부-접수 버튼 사이 영역으로 제한된 시간 조절 감지기"""
    
    def __init__(self):
        print("🔄 ROI 제한 시간 조절기 초기화...")
        
        # EasyOCR 초기화
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
        
        # 결과 저장 폴더
        self.output_dir = Path("test_results/roi_limited_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ 초기화 완료!")
    
    def capture_screen(self):
        """화면 캡처"""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                screenshot = sct.grab(monitor)
                img_array = np.array(screenshot)
                return cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
        except Exception as e:
            print(f"❌ 화면 캡처 실패: {e}")
            return None
    
    def find_boundary_buttons(self, image):
        """
        경계 버튼 (거부, 접수) 찾기
        
        Returns:
            dict: 거부/접수 버튼 정보
        """
        print("🔍 경계 버튼 (거부/접수) 찾기...")
        
        # EasyOCR로 모든 텍스트 감지
        results = self.reader.readtext(image, paragraph=False)
        
        reject_button = None
        accept_button = None
        
        # 거부/접수 관련 키워드
        reject_keywords = ['거부', '거절', '취소', '반려']
        accept_keywords = ['접수', '수락', '확인', '승인']
        
        for bbox, text, confidence in results:
            if confidence < 0.7:  # 높은 신뢰도만
                continue
                
            text_clean = text.strip()
            
            # 거부 버튼 찾기
            for keyword in reject_keywords:
                if keyword in text_clean:
                    center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                    center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                    width = int(bbox[2][0] - bbox[0][0])
                    height = int(bbox[2][1] - bbox[0][1])
                    
                    reject_button = {
                        'text': text_clean,
                        'center': (center_x, center_y),
                        'bbox': bbox,
                        'left': int(bbox[0][0]),
                        'right': int(bbox[2][0]),
                        'top': int(bbox[0][1]),
                        'bottom': int(bbox[2][1]),
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    }
                    print(f"❌ 거부 버튼 발견: '{text_clean}' at ({center_x}, {center_y})")
                    break
            
            # 접수 버튼 찾기
            for keyword in accept_keywords:
                if keyword in text_clean:
                    center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                    center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                    width = int(bbox[2][0] - bbox[0][0])
                    height = int(bbox[2][1] - bbox[0][1])
                    
                    accept_button = {
                        'text': text_clean,
                        'center': (center_x, center_y),
                        'bbox': bbox,
                        'left': int(bbox[0][0]),
                        'right': int(bbox[2][0]),
                        'top': int(bbox[0][1]),
                        'bottom': int(bbox[2][1]),
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    }
                    print(f"✅ 접수 버튼 발견: '{text_clean}' at ({center_x}, {center_y})")
                    break
        
        return {
            'reject': reject_button,
            'accept': accept_button
        }
    
    def calculate_roi_between_buttons(self, reject_button, accept_button, image_shape):
        """
        거부/접수 버튼 사이의 ROI 영역 계산
        
        Returns:
            tuple: (roi_coordinates, roi_image)
        """
        if not reject_button or not accept_button:
            print("⚠️ 경계 버튼 중 하나를 찾지 못했습니다")
            return None, None
        
        height, width = image_shape[:2]
        
        # ROI 경계 계산
        roi_left = reject_button['right'] + 10    # 거부 버튼 오른쪽에서 10px
        roi_right = accept_button['left'] - 10     # 접수 버튼 왼쪽에서 10px
        
        # 세로 영역은 두 버튼을 모두 포함하도록
        roi_top = min(reject_button['top'], accept_button['top']) - 30
        roi_bottom = max(reject_button['bottom'], accept_button['bottom']) + 30
        
        # 경계 검증
        roi_left = max(0, roi_left)
        roi_right = min(width, roi_right)
        roi_top = max(0, roi_top)
        roi_bottom = min(height, roi_bottom)
        
        # ROI 유효성 검사
        if roi_right <= roi_left or roi_bottom <= roi_top:
            print("❌ ROI 영역이 유효하지 않습니다")
            return None, None
        
        roi_coords = (roi_left, roi_top, roi_right, roi_bottom)
        roi_width = roi_right - roi_left
        roi_height = roi_bottom - roi_top
        
        print(f"📐 ROI 영역 계산됨:")
        print(f"   좌표: ({roi_left}, {roi_top}) → ({roi_right}, {roi_bottom})")
        print(f"   크기: {roi_width} x {roi_height}")
        print(f"   전체 화면 대비: {roi_width * roi_height / (width * height) * 100:.1f}%")
        
        return roi_coords, roi_coords
    
    def detect_time_controls_in_roi(self, image, roi_coords):
        """
        ROI 영역 내에서만 시간 조절 요소 감지
        
        Args:
            image: 전체 이미지
            roi_coords: (left, top, right, bottom)
        """
        if not roi_coords:
            return {}
        
        roi_left, roi_top, roi_right, roi_bottom = roi_coords
        
        # ROI 영역 추출
        roi_image = image[roi_top:roi_bottom, roi_left:roi_right]
        
        print(f"🔬 ROI 내에서 시간 조절 요소 감지 중...")
        
        # ROI에서 OCR 실행 (훨씬 빠르고 정확)
        results = self.reader.readtext(roi_image, paragraph=False)
        
        time_displays = []
        plus_buttons = []
        minus_buttons = []
        all_detections = []
        
        for bbox, text, confidence in results:
            if confidence < 0.3:  # ROI 내에서는 좀 더 관대하게
                continue
            
            text_clean = text.strip()
            
            # ROI 내 좌표를 전체 이미지 좌표로 변환
            local_center_x = int((bbox[0][0] + bbox[2][0]) / 2)
            local_center_y = int((bbox[0][1] + bbox[2][1]) / 2)
            global_center_x = local_center_x + roi_left
            global_center_y = local_center_y + roi_top
            
            detection = {
                'text': text_clean,
                'confidence': confidence,
                'local_center': (local_center_x, local_center_y),
                'global_center': (global_center_x, global_center_y),
                'bbox': bbox
            }
            all_detections.append(detection)
            
            print(f"📝 ROI 내 텍스트: '{text_clean}' (신뢰도: {confidence:.3f})")
            
            # 시간 표시 감지
            if '분' in text_clean and any(c.isdigit() for c in text_clean):
                time_displays.append(detection)
                print(f"⏰ 시간 표시: '{text_clean}'")
            
            # + 버튼 감지
            if '+' in text_clean or '＋' in text_clean:
                plus_buttons.append(detection)
                print(f"➕ + 버튼: '{text_clean}'")
            
            # - 버튼 감지  
            if '-' in text_clean or '－' in text_clean or '—' in text_clean:
                minus_buttons.append(detection)
                print(f"➖ - 버튼: '{text_clean}'")
        
        return {
            'time_displays': time_displays,
            'plus_buttons': plus_buttons,
            'minus_buttons': minus_buttons,
            'all_detections': all_detections,
            'roi_coords': roi_coords,
            'roi_image': roi_image
        }
    
    def estimate_button_positions_from_time(self, time_displays, roi_coords):
        """
        시간 표시를 기반으로 +/- 버튼 위치 추정
        """
        if not time_displays:
            return []
        
        estimated_positions = []
        
        for time_display in time_displays:
            global_x, global_y = time_display['global_center']
            
            # 시간 텍스트 길이에 따른 오프셋 조정
            text_length = len(time_display['text'])
            base_offset = 50 + (text_length * 3)  # 텍스트 길이에 따라 조정
            
            # - 버튼 (왼쪽)
            minus_x = global_x - base_offset
            minus_pos = {
                'type': 'minus',
                'estimated_position': (minus_x, global_y),
                'reference_time': time_display['text'],
                'method': 'time_based_estimation'
            }
            
            # + 버튼 (오른쪽)  
            plus_x = global_x + base_offset
            plus_pos = {
                'type': 'plus',
                'estimated_position': (plus_x, global_y),
                'reference_time': time_display['text'],
                'method': 'time_based_estimation'
            }
            
            estimated_positions.extend([minus_pos, plus_pos])
            
            print(f"📍 '{time_display['text']}' 기준 추정:")
            print(f"   - 버튼: ({minus_x}, {global_y})")
            print(f"   + 버튼: ({plus_x}, {global_y})")
        
        return estimated_positions
    
    def visualize_results(self, image, boundary_buttons, roi_results, estimated_positions, timestamp):
        """결과 시각화"""
        annotated = image.copy()
        
        # 경계 버튼 표시
        if boundary_buttons['reject']:
            bbox = boundary_buttons['reject']['bbox']
            pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 0, 255), 3)  # 빨간색
            center = boundary_buttons['reject']['center']
            cv2.putText(annotated, "REJECT", (center[0]-30, center[1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if boundary_buttons['accept']:
            bbox = boundary_buttons['accept']['bbox']
            pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)  # 초록색
            center = boundary_buttons['accept']['center']
            cv2.putText(annotated, "ACCEPT", (center[0]-30, center[1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # ROI 영역 표시
        if roi_results and 'roi_coords' in roi_results:
            roi_left, roi_top, roi_right, roi_bottom = roi_results['roi_coords']
            cv2.rectangle(annotated, (roi_left, roi_top), (roi_right, roi_bottom), 
                         (255, 255, 0), 3)  # 노란색 ROI
            cv2.putText(annotated, "ROI", (roi_left, roi_top-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # ROI 내 감지된 요소들 표시
        if roi_results:
            for time_display in roi_results.get('time_displays', []):
                center = time_display['global_center']
                cv2.circle(annotated, center, 15, (255, 0, 255), -1)  # 보라색
                cv2.putText(annotated, "TIME", (center[0]-20, center[1]+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            for plus_btn in roi_results.get('plus_buttons', []):
                center = plus_btn['global_center']
                cv2.circle(annotated, center, 12, (0, 255, 255), -1)  # 시안색
                cv2.putText(annotated, "+", (center[0]-5, center[1]+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            for minus_btn in roi_results.get('minus_buttons', []):
                center = minus_btn['global_center']
                cv2.circle(annotated, center, 12, (255, 128, 0), -1)  # 주황색
                cv2.putText(annotated, "-", (center[0]-5, center[1]+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
        
        # 추정된 위치 표시
        for pos in estimated_positions:
            center = pos['estimated_position']
            if pos['type'] == 'plus':
                cv2.drawMarker(annotated, center, (0, 255, 255), 
                              cv2.MARKER_CROSS, 20, 3)
                cv2.putText(annotated, "EST+", (center[0]-20, center[1]-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.drawMarker(annotated, center, (255, 128, 0), 
                              cv2.MARKER_CROSS, 20, 3)
                cv2.putText(annotated, "EST-", (center[0]-20, center[1]-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
        
        # 결과 이미지 저장
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        annotated_path = self.output_dir / f"roi_limited_result_{timestamp}.png"
        cv2.imwrite(str(annotated_path), annotated_bgr)
        
        print(f"📊 시각화 결과 저장: {annotated_path}")
        return str(annotated_path)
    
    def run_roi_limited_test(self):
        """ROI 제한 테스트 실행"""
        print("🚀 ROI 제한 시간 조절 버튼 감지 테스트")
        print("거부-접수 버튼 사이 영역으로만 검색 제한")
        print("=" * 60)
        
        # 화면 캡처
        screenshot = self.capture_screen()
        if screenshot is None:
            return False
        
        timestamp = int(time.time())
        
        # 원본 이미지 저장
        original_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        original_path = self.output_dir / f"original_roi_test_{timestamp}.png"
        cv2.imwrite(str(original_path), original_bgr)
        print(f"📷 원본 저장: {original_path}")
        
        # 1단계: 경계 버튼 찾기
        print(f"\n1️⃣ 경계 버튼 찾기...")
        boundary_buttons = self.find_boundary_buttons(screenshot)
        
        if not boundary_buttons['reject'] and not boundary_buttons['accept']:
            print("❌ 거부/접수 버튼을 찾지 못했습니다")
            return False
        
        # 2단계: ROI 계산
        print(f"\n2️⃣ ROI 영역 계산...")
        roi_coords, _ = self.calculate_roi_between_buttons(
            boundary_buttons['reject'], 
            boundary_buttons['accept'], 
            screenshot.shape
        )
        
        if not roi_coords:
            print("❌ ROI 계산 실패")
            return False
        
        # 3단계: ROI 내에서만 시간 조절 요소 감지
        print(f"\n3️⃣ ROI 제한 감지...")
        roi_results = self.detect_time_controls_in_roi(screenshot, roi_coords)
        
        # 4단계: 시간 기반 위치 추정
        print(f"\n4️⃣ 버튼 위치 추정...")
        estimated_positions = self.estimate_button_positions_from_time(
            roi_results.get('time_displays', []), roi_coords
        )
        
        # 5단계: 결과 시각화
        print(f"\n5️⃣ 결과 시각화...")
        visualized_path = self.visualize_results(
            screenshot, boundary_buttons, roi_results, estimated_positions, timestamp
        )
        
        # 결과 요약
        print(f"\n" + "=" * 60)
        print("🎯 테스트 결과")
        print("=" * 60)
        
        print(f"✅ 경계 버튼:")
        print(f"   거부: {'발견됨' if boundary_buttons['reject'] else '없음'}")
        print(f"   접수: {'발견됨' if boundary_buttons['accept'] else '없음'}")
        
        if roi_results:
            roi_area = (roi_coords[2] - roi_coords[0]) * (roi_coords[3] - roi_coords[1])
            total_area = screenshot.shape[0] * screenshot.shape[1] 
            efficiency = (1 - roi_area / total_area) * 100
            
            print(f"📐 ROI 효율성: {efficiency:.1f}% 검색 영역 감소")
            print(f"🔍 ROI 내 감지 결과:")
            print(f"   시간 표시: {len(roi_results.get('time_displays', []))}개")
            print(f"   + 버튼: {len(roi_results.get('plus_buttons', []))}개")
            print(f"   - 버튼: {len(roi_results.get('minus_buttons', []))}개")
            print(f"   총 감지: {len(roi_results.get('all_detections', []))}개")
            
        print(f"📍 추정된 버튼 위치: {len(estimated_positions)}개")
        
        print(f"\n📁 생성된 파일:")
        print(f"   - 원본: {original_path}")
        print(f"   - 시각화: {visualized_path}")
        
        success = len(estimated_positions) > 0
        
        if success:
            print(f"\n🎉 ROI 제한 방식이 효과적입니다!")
            print("   → 거부-접수 버튼 사이 영역으로 시간 조절 가능")
        else:
            print(f"\n⚠️ 시간 조절 요소를 찾지 못했습니다")
            print("   → 추가 최적화가 필요할 수 있습니다")
        
        return success

def main():
    """메인 실행"""
    print("🎯 ROI 제한 시간 조절 감지 테스트")
    print("거부-접수 버튼 사이 영역으로만 검색하는 똑똑한 방법")
    print()
    
    input("배달앱에서 거부, 시간조절, 접수 버튼이 모두 보이는 화면으로 이동 후 Enter... ")
    
    try:
        controller = ROILimitedTimeController()
        success = controller.run_roi_limited_test()
        
        if success:
            print(f"\n🎉 검증 완료! 이 방식이 매우 유망합니다!")
            print("   → DeepOrder에 바로 적용 가능")
        else:
            print(f"\n🤔 추가 조정이 필요할 수 있습니다")
        
    except Exception as e:
        print(f"💥 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
