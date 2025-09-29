#!/usr/bin/env python3
"""
🎯 실제 쿠팡이츠 이미지에서 한글 "접수" 버튼 감지 테스트
사용자 제공 스크린샷에서 실제 한글 텍스트 버튼 찾기
"""

import cv2
import numpy as np
import easyocr
import os
from pathlib import Path
import time
import json
import mss

class RealKoreanButtonTester:
    """실제 한글 버튼 감지 테스터"""
    
    def __init__(self):
        print("🔄 EasyOCR 한글 모드 초기화 중...")
        start_time = time.time()
        
        # EasyOCR Reader 초기화 (한국어 우선 + 영어)
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
        
        init_time = time.time() - start_time
        print(f"✅ EasyOCR 초기화 완료! ({init_time:.2f}초)")
        
        # 결과 저장 폴더
        self.output_dir = Path("test_results/korean_button_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def capture_current_screen(self):
        """현재 화면 캡처 (실제 쿠팡이츠 화면)"""
        print("📷 현재 화면 캡처 중...")
        
        try:
            with mss.mss() as sct:
                # 전체 화면 캡처
                monitor = sct.monitors[0]  # 기본 모니터
                screenshot = sct.grab(monitor)
                
                # numpy 배열로 변환
                img_array = np.array(screenshot)
                
                # BGRA에서 RGB로 변환
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
                
                print(f"   화면 크기: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
                return img_rgb
                
        except Exception as e:
            print(f"❌ 화면 캡처 실패: {e}")
            return None
    
    def detect_korean_buttons(self, image):
        """
        한글 텍스트 버튼 감지
        
        Args:
            image: 입력 이미지 (numpy array)
            
        Returns:
            dict: 감지된 한글 버튼들의 정보
        """
        print("🔍 한글 텍스트 감지 중...")
        start_time = time.time()
        
        # EasyOCR 실행 (한글 최우선)
        results = self.reader.readtext(image, paragraph=False)
        
        processing_time = time.time() - start_time
        print(f"⚡ OCR 처리 완료: {processing_time:.3f}초")
        
        detected_buttons = {}
        all_korean_texts = []
        
        # 한글 키워드 정의
        button_keywords = {
            'accept': ['접수', '수락', '확인', '승인', '받기'],
            'reject': ['거부', '거절', '취소', '반려', '닫기'],
            'prepare': ['준비', '조리', '완료'],
            'delivery': ['배달', '픽업', '수거']
        }
        
        # 결과 분석
        for i, (bbox, text, confidence) in enumerate(results):
            text_clean = text.strip()
            print(f"📝 감지된 텍스트 {i+1}: '{text_clean}' (신뢰도: {confidence:.3f})")
            
            # 한글이 포함된 텍스트 별도 저장
            if any('\uac00' <= char <= '\ud7af' for char in text_clean):
                all_korean_texts.append({
                    'text': text_clean,
                    'confidence': confidence,
                    'bbox': bbox
                })
                print(f"🇰🇷 한글 텍스트 발견: '{text_clean}'")
            
            # 버튼 키워드 매칭
            for button_type, keywords in button_keywords.items():
                for keyword in keywords:
                    if keyword in text_clean and confidence > 0.5:
                        center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                        center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                        
                        button_info = {
                            'type': button_type,
                            'text': text_clean,
                            'confidence': confidence,
                            'bbox': bbox,
                            'center': (center_x, center_y),
                            'keyword_matched': keyword
                        }
                        
                        detected_buttons[button_type] = button_info
                        print(f"🎯 {button_type} 버튼 발견! '{text_clean}' at ({center_x}, {center_y})")
                        break
        
        return {
            'buttons': detected_buttons,
            'all_results': results,
            'korean_texts': all_korean_texts,
            'processing_time': processing_time
        }
    
    def save_button_regions(self, image, detection_results):
        """감지된 버튼 영역들을 개별 이미지로 저장"""
        saved_files = []
        
        for button_type, button_info in detection_results['buttons'].items():
            bbox = button_info['bbox']
            
            # 바운딩 박스에서 좌표 추출
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # 여백 추가 (버튼을 좀 더 크게)
            margin = 30
            h, w = image.shape[:2]
            
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            # 버튼 영역 추출
            button_region = image[y_min:y_max, x_min:x_max]
            
            # BGR로 변환해서 저장 (OpenCV 형식)
            button_bgr = cv2.cvtColor(button_region, cv2.COLOR_RGB2BGR)
            
            # 파일명 생성 (한글 텍스트 포함)
            timestamp = int(time.time())
            safe_text = button_info['text'].replace(' ', '_').replace('/', '_')
            filename = f"{button_type}_{safe_text}_{timestamp}.png"
            filepath = self.output_dir / filename
            
            # 이미지 저장
            cv2.imwrite(str(filepath), button_bgr)
            saved_files.append(str(filepath))
            
            print(f"💾 {button_type} 버튼 저장: {filepath}")
            print(f"   텍스트: '{button_info['text']}'")
            print(f"   좌표: {button_info['center']}")
            print(f"   영역: ({x_min}, {y_min}) - ({x_max}, {y_max})")
        
        return saved_files
    
    def create_annotated_screenshot(self, image, detection_results):
        """감지 결과를 표시한 주석 스크린샷 생성"""
        annotated = image.copy()
        
        # 모든 감지된 텍스트에 바운딩 박스 그리기
        for bbox, text, confidence in detection_results['all_results']:
            # 바운딩 박스
            pts = np.array(bbox, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # 한글이 포함된 텍스트는 빨간색, 영어는 파란색
            if any('\uac00' <= char <= '\ud7af' for char in text):
                color = (255, 0, 0)  # 빨간색 (한글)
                thickness = 3
            else:
                color = (0, 100, 255)  # 주황색 (영어/숫자)
                thickness = 2
            
            cv2.polylines(annotated, [pts], True, color, thickness)
            
            # 신뢰도 표시
            if confidence > 0.5:
                center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                
                label = f"{confidence:.2f}"
                cv2.putText(annotated, label, 
                           (center_x-20, center_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 감지된 버튼들에 특별 표시
        for button_type, button_info in detection_results['buttons'].items():
            center = button_info['center']
            
            # 큰 원으로 중심점 강조
            if button_type == 'accept':
                cv2.circle(annotated, center, 20, (0, 255, 0), -1)  # 초록색
                label_color = (0, 255, 0)
            elif button_type == 'reject':
                cv2.circle(annotated, center, 20, (0, 0, 255), -1)  # 빨간색
                label_color = (0, 0, 255)
            else:
                cv2.circle(annotated, center, 15, (255, 255, 0), -1)  # 노란색
                label_color = (255, 255, 0)
            
            # 버튼 타입 라벨
            cv2.putText(annotated, button_type.upper(), 
                       (center[0]-30, center[1]+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 3)
        
        # 주석 이미지 저장
        timestamp = int(time.time())
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        annotated_path = self.output_dir / f"annotated_korean_detection_{timestamp}.png"
        cv2.imwrite(str(annotated_path), annotated_bgr)
        
        print(f"📊 주석 스크린샷 저장: {annotated_path}")
        return str(annotated_path)
    
    def run_korean_button_test(self):
        """실제 한글 버튼 감지 테스트 실행"""
        print("🚀 실제 쿠팡이츠 한글 접수 버튼 감지 테스트!")
        print("=" * 70)
        
        # 1. 현재 화면 캡처
        print("1️⃣ 현재 화면 캡처...")
        screenshot = self.capture_current_screen()
        
        if screenshot is None:
            print("❌ 화면 캡처 실패")
            return False
        
        # 원본 스크린샷 저장
        timestamp = int(time.time())
        original_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        original_path = self.output_dir / f"original_coupang_screenshot_{timestamp}.png"
        cv2.imwrite(str(original_path), original_bgr)
        print(f"   원본 저장: {original_path}")
        
        # 2. 한글 텍스트 감지
        print("\n2️⃣ 한글 텍스트 감지...")
        detection_results = self.detect_korean_buttons(screenshot)
        
        # 3. 결과 분석
        print(f"\n3️⃣ 감지 결과 분석...")
        print(f"   총 감지된 텍스트: {len(detection_results['all_results'])}개")
        print(f"   한글 텍스트: {len(detection_results['korean_texts'])}개")
        print(f"   감지된 버튼: {len(detection_results['buttons'])}개")
        
        # 한글 텍스트들 출력
        if detection_results['korean_texts']:
            print("\n🇰🇷 발견된 한글 텍스트들:")
            for korean in detection_results['korean_texts']:
                print(f"   - '{korean['text']}' (신뢰도: {korean['confidence']:.3f})")
        
        # 4. 접수 버튼 확인
        if 'accept' in detection_results['buttons']:
            print(f"\n4️⃣ ✅ 접수 버튼 발견!")
            accept_info = detection_results['buttons']['accept']
            print(f"   텍스트: '{accept_info['text']}'")
            print(f"   매칭 키워드: '{accept_info['keyword_matched']}'")
            print(f"   좌표: {accept_info['center']}")
            print(f"   신뢰도: {accept_info['confidence']:.1%}")
        else:
            print(f"\n4️⃣ ❌ 접수 버튼을 찾지 못했습니다")
            print("   감지된 한글 텍스트들을 확인해보세요:")
            for korean in detection_results['korean_texts'][:5]:  # 상위 5개만
                print(f"   - '{korean['text']}'")
        
        # 5. 버튼 영역 추출 및 저장
        print(f"\n5️⃣ 버튼 영역 추출...")
        saved_files = self.save_button_regions(screenshot, detection_results)
        
        # 6. 주석 이미지 생성
        print(f"\n6️⃣ 주석 이미지 생성...")
        annotated_path = self.create_annotated_screenshot(screenshot, detection_results)
        
        # 7. 최종 결과 요약
        print("\n" + "=" * 70)
        print("🎯 테스트 결과 요약")
        print("=" * 70)
        
        success = 'accept' in detection_results['buttons']
        
        if success:
            accept_info = detection_results['buttons']['accept']
            print(f"✅ 한글 접수 버튼 감지 성공!")
            print(f"   📍 위치: {accept_info['center']}")
            print(f"   📝 텍스트: '{accept_info['text']}'") 
            print(f"   🎯 신뢰도: {accept_info['confidence']:.1%}")
            print(f"   🔑 키워드: '{accept_info['keyword_matched']}'")
        else:
            print("❌ 접수 버튼 감지 실패")
            if detection_results['korean_texts']:
                print("   하지만 다른 한글 텍스트들은 감지됨:")
                for korean in detection_results['korean_texts'][:3]:
                    print(f"   - '{korean['text']}'")
        
        print(f"\n📁 생성된 파일들:")
        print(f"   - 원본: {original_path}")
        print(f"   - 주석: {annotated_path}")
        for file_path in saved_files:
            print(f"   - 버튼: {file_path}")
        
        print(f"\n⚡ 처리 시간: {detection_results['processing_time']:.3f}초")
        
        if success:
            print("🎉 테스트 성공!")
        else:
            print("⚠️  접수 버튼을 찾지 못했지만 다른 텍스트는 감지됨")
        
        return success

def main():
    """메인 함수"""
    print("🇰🇷 실제 한글 접수 버튼 감지 테스트")
    print("쿠팡이츠나 배달의민족 화면에서 실제 '접수' 버튼 찾기")
    print("\n⚠️  주의: 테스트하려는 앱 화면이 보이는 상태로 실행하세요!")
    
    input("\n화면에 배달앱을 띄운 후 Enter를 눌러주세요... ")
    
    try:
        tester = RealKoreanButtonTester()
        success = tester.run_korean_button_test()
        
        if success:
            print("\n🎯 완벽한 성공!")
            print("   DeepOrder에서 이 결과를 바로 사용할 수 있습니다!")
            return 0
        else:
            print("\n🤔 일부 성공!")
            print("   텍스트는 감지되었지만 '접수' 키워드를 찾지 못했습니다.")
            print("   감지된 텍스트들을 확인해서 키워드를 추가하면 됩니다.")
            return 1
            
    except Exception as e:
        print(f"\n💥 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit(main())
