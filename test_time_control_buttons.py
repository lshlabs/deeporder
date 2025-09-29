#!/usr/bin/env python3
"""
🎯 배달앱 시간 조절 버튼 감지 테스트
"— 20~25분 +" 형태의 시간 조절 UI에서 - 버튼과 + 버튼을 찾는 방법들 비교
"""

import cv2
import numpy as np
import easyocr
import mss
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional

class TimeControlButtonTester:
    """시간 조절 버튼 감지 테스터"""
    
    def __init__(self):
        print("🔄 시간 조절 버튼 테스터 초기화...")
        
        # EasyOCR 초기화
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
        
        # 결과 저장 폴더
        self.output_dir = Path("test_results/time_control_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ 초기화 완료!")
    
    def capture_screen(self):
        """현재 화면 캡처"""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                screenshot = sct.grab(monitor)
                img_array = np.array(screenshot)
                return cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
        except Exception as e:
            print(f"❌ 화면 캡처 실패: {e}")
            return None
    
    def method1_easyocr_symbols(self, image):
        """
        방법 1: EasyOCR로 직접 +, - 기호 감지 시도
        """
        print("\n📝 방법 1: EasyOCR 기호 직접 감지")
        
        start_time = time.time()
        results = self.reader.readtext(image, paragraph=False)
        processing_time = time.time() - start_time
        
        plus_buttons = []
        minus_buttons = []
        time_displays = []
        
        for bbox, text, confidence in results:
            text_clean = text.strip()
            
            if confidence > 0.3:  # 기호는 신뢰도가 낮을 수 있으므로
                center = ((bbox[0][0] + bbox[2][0]) // 2, (bbox[0][1] + bbox[2][1]) // 2)
                
                # + 기호 감지
                if '+' in text_clean or '十' in text_clean or 'ㅗ' in text_clean:
                    plus_buttons.append({
                        'center': center,
                        'text': text_clean,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    print(f"➕ + 버튼 발견: '{text_clean}' (신뢰도: {confidence:.3f})")
                
                # - 기호 감지
                if '-' in text_clean or '—' in text_clean or 'ㅡ' in text_clean or '_' in text_clean:
                    minus_buttons.append({
                        'center': center,
                        'text': text_clean,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    print(f"➖ - 버튼 발견: '{text_clean}' (신뢰도: {confidence:.3f})")
                
                # 시간 표시 감지 (분)
                if '분' in text_clean or 'min' in text_clean.lower():
                    if any(char.isdigit() for char in text_clean):
                        time_displays.append({
                            'center': center,
                            'text': text_clean,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        print(f"⏰ 시간 표시 발견: '{text_clean}' (신뢰도: {confidence:.3f})")
        
        return {
            'plus_buttons': plus_buttons,
            'minus_buttons': minus_buttons,
            'time_displays': time_displays,
            'processing_time': processing_time,
            'method': 'EasyOCR 기호 직접 감지'
        }
    
    def method2_hybrid_approach(self, image):
        """
        방법 2: 하이브리드 접근 - 시간 표시를 찾고 주변의 버튼 영역 추정
        """
        print("\n🔬 방법 2: 시간 표시 기반 버튼 위치 추정")
        
        start_time = time.time()
        
        # 1단계: EasyOCR로 시간 표시 찾기
        results = self.reader.readtext(image, paragraph=False)
        
        time_regions = []
        for bbox, text, confidence in results:
            text_clean = text.strip()
            
            # 시간 패턴 찾기 (예: "20~25분", "10-15분", "15분" 등)
            if ('분' in text_clean or 'min' in text_clean.lower()) and confidence > 0.7:
                if any(char.isdigit() for char in text_clean):
                    center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                    center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                    width = int(bbox[2][0] - bbox[0][0])
                    height = int(bbox[2][1] - bbox[0][1])
                    
                    time_regions.append({
                        'text': text_clean,
                        'center': (center_x, center_y),
                        'bbox': bbox,
                        'confidence': confidence,
                        'width': width,
                        'height': height
                    })
                    print(f"⏰ 시간 영역 발견: '{text_clean}' at ({center_x}, {center_y})")
        
        # 2단계: 각 시간 영역 주변에서 - 및 + 버튼 위치 추정
        estimated_buttons = []
        
        for time_region in time_regions:
            center_x, center_y = time_region['center']
            width = time_region['width']
            
            # - 버튼 추정 위치 (시간 표시 왼쪽)
            minus_x = center_x - width//2 - 60  # 시간 표시 왼쪽으로 60px
            minus_y = center_y
            
            # + 버튼 추정 위치 (시간 표시 오른쪽)
            plus_x = center_x + width//2 + 60   # 시간 표시 오른쪽으로 60px
            plus_y = center_y
            
            estimated_buttons.extend([
                {
                    'type': 'minus',
                    'center': (minus_x, minus_y),
                    'estimated': True,
                    'time_reference': time_region['text']
                },
                {
                    'type': 'plus', 
                    'center': (plus_x, plus_y),
                    'estimated': True,
                    'time_reference': time_region['text']
                }
            ])
            
            print(f"📍 추정된 - 버튼: ({minus_x}, {minus_y})")
            print(f"📍 추정된 + 버튼: ({plus_x}, {plus_y})")
        
        processing_time = time.time() - start_time
        
        return {
            'time_regions': time_regions,
            'estimated_buttons': estimated_buttons,
            'processing_time': processing_time,
            'method': '시간 기반 위치 추정'
        }
    
    def method3_color_shape_detection(self, image):
        """
        방법 3: 색상 및 모양 기반 버튼 감지 (OpenCV)
        """
        print("\n🎨 방법 3: 색상/모양 기반 버튼 감지")
        
        start_time = time.time()
        
        # BGR로 변환 (OpenCV용)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # 원형 버튼 감지 (HoughCircles)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=50
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 원형 영역 추출해서 내용 분석
                roi = gray[max(0, y-r):min(gray.shape[0], y+r), 
                          max(0, x-r):min(gray.shape[1], x+r)]
                
                # 간단한 패턴 매칭으로 +/- 구분 시도
                # (실제로는 더 정교한 분석 필요)
                
                detected_circles.append({
                    'center': (x, y),
                    'radius': r,
                    'type': 'unknown_button'
                })
                print(f"⭕ 원형 버튼 발견: ({x}, {y}) 반지름 {r}")
        
        processing_time = time.time() - start_time
        
        return {
            'circles': detected_circles,
            'processing_time': processing_time,
            'method': '색상/모양 기반 감지'
        }
    
    def comprehensive_test(self):
        """모든 방법을 종합적으로 테스트"""
        print("🚀 시간 조절 버튼 감지 종합 테스트")
        print("=" * 60)
        
        # 화면 캡처
        screenshot = self.capture_screen()
        if screenshot is None:
            return
        
        # 원본 이미지 저장
        timestamp = int(time.time())
        original_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        original_path = self.output_dir / f"original_time_control_{timestamp}.png"
        cv2.imwrite(str(original_path), original_bgr)
        print(f"📷 원본 저장: {original_path}")
        
        # 세 가지 방법으로 테스트
        results = {}
        
        # 방법 1: EasyOCR 직접 감지
        results['method1'] = self.method1_easyocr_symbols(screenshot)
        
        # 방법 2: 하이브리드 접근
        results['method2'] = self.method2_hybrid_approach(screenshot)
        
        # 방법 3: 색상/모양 기반
        results['method3'] = self.method3_color_shape_detection(screenshot)
        
        # 결과 분석 및 시각화
        self.analyze_and_visualize_results(screenshot, results, timestamp)
        
        return results
    
    def analyze_and_visualize_results(self, image, results, timestamp):
        """결과 분석 및 시각화"""
        print(f"\n📊 결과 분석")
        print("=" * 60)
        
        # 주석 이미지 생성
        annotated = image.copy()
        
        # 방법 1 결과 표시 (빨간색)
        method1 = results['method1']
        for btn in method1['plus_buttons']:
            cv2.circle(annotated, btn['center'], 25, (255, 0, 0), 3)
            cv2.putText(annotated, "M1:+", 
                       (btn['center'][0]-20, btn['center'][1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        for btn in method1['minus_buttons']:
            cv2.circle(annotated, btn['center'], 25, (255, 0, 0), 3)
            cv2.putText(annotated, "M1:-", 
                       (btn['center'][0]-20, btn['center'][1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 방법 2 결과 표시 (초록색)
        method2 = results['method2']
        for btn in method2['estimated_buttons']:
            color = (0, 255, 0)
            symbol = "M2:+" if btn['type'] == 'plus' else "M2:-"
            cv2.circle(annotated, btn['center'], 20, color, 3)
            cv2.putText(annotated, symbol,
                       (btn['center'][0]-20, btn['center'][1]+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 시간 표시 영역 표시
        for time_region in method2['time_regions']:
            bbox = time_region['bbox']
            pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)
        
        # 방법 3 결과 표시 (파란색)
        method3 = results['method3'] 
        for circle in method3['circles']:
            cv2.circle(annotated, circle['center'], circle['radius'], (0, 0, 255), 3)
            cv2.putText(annotated, "M3",
                       (circle['center'][0]-10, circle['center'][1]+circle['radius']+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 주석 이미지 저장
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        annotated_path = self.output_dir / f"annotated_time_control_{timestamp}.png"
        cv2.imwrite(str(annotated_path), annotated_bgr)
        
        # 결과 요약 출력
        print(f"📋 방법별 결과:")
        print(f"   방법 1 (EasyOCR 직접): + {len(method1['plus_buttons'])}개, - {len(method1['minus_buttons'])}개, 시간 {len(method1['time_displays'])}개")
        print(f"   방법 2 (하이브리드): 시간영역 {len(method2['time_regions'])}개, 추정버튼 {len(method2['estimated_buttons'])}개")
        print(f"   방법 3 (색상/모양): 원형버튼 {len(method3['circles'])}개")
        
        print(f"\n📁 생성된 파일:")
        print(f"   - 주석 이미지: {annotated_path}")
        
        # 최적 방법 추천
        self.recommend_best_approach(results)
    
    def recommend_best_approach(self, results):
        """최적 접근 방법 추천"""
        print(f"\n💡 추천 방법:")
        
        method1 = results['method1']
        method2 = results['method2']
        method3 = results['method3']
        
        # 방법 2 (하이브리드)가 가장 안정적일 가능성이 높음
        if len(method2['time_regions']) > 0:
            print("🥇 **방법 2 (하이브리드)** 추천!")
            print("   📍 시간 표시를 EasyOCR로 찾고, 그 주변에 +/- 버튼 위치 추정")
            print("   ✅ 장점: 안정적, 정확한 위치, 시간 변화 감지 가능")
            print("   📝 구현 방법:")
            print("      1. '20~25분' 같은 시간 텍스트를 EasyOCR로 찾기")
            print("      2. 시간 텍스트 왼쪽/오른쪽으로 일정 거리에 +/- 버튼 있다고 가정")
            print("      3. 해당 위치를 클릭하고 시간 변화 확인")
        
        elif len(method1['plus_buttons']) > 0 or len(method1['minus_buttons']) > 0:
            print("🥈 **방법 1 (EasyOCR 직접)** 사용 가능!")
            print("   📝 +/- 기호를 직접 감지")
            
        elif len(method3['circles']) > 0:
            print("🥉 **방법 3 (색상/모양)** 보조적 사용")
            print("   📝 원형 버튼 모양으로 후보 찾기")
        
        else:
            print("⚠️ 모든 방법에서 버튼을 찾지 못했습니다")
            print("   💡 해결책:")
            print("      1. ROI 영역을 시간 조절 부분으로 좁히기")
            print("      2. 이미지 전처리로 대비 향상")
            print("      3. 신뢰도 임계값 조정")

def main():
    """메인 실행"""
    print("🎯 배달앱 시간 조절 버튼 감지 테스트")
    print("'— 20~25분 +' 형태의 시간 조절 UI 분석")
    print()
    
    input("배달앱 시간 조절 화면을 띄운 후 Enter를 눌러주세요... ")
    
    try:
        tester = TimeControlButtonTester()
        results = tester.comprehensive_test()
        
        print(f"\n🎉 테스트 완료!")
        print("결과 이미지를 확인해서 어떤 방법이 가장 효과적인지 확인하세요.")
        
    except Exception as e:
        print(f"💥 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
