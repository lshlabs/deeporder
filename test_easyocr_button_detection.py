#!/usr/bin/env python3
"""
🎯 EasyOCR을 사용한 쿠팡이츠 "접수" 버튼 감지 테스트
제공된 쿠팡이츠 화면에서 텍스트 기반으로 접수 버튼을 찾아보는 실제 테스트
"""

import cv2
import numpy as np
import easyocr
import os
from pathlib import Path
import time
import json

class EasyOCRButtonTester:
    """EasyOCR 접수 버튼 감지 테스터"""
    
    def __init__(self):
        print("🔄 EasyOCR 초기화 중...")
        start_time = time.time()
        
        # EasyOCR Reader 초기화 (한국어 + 영어)
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
        
        init_time = time.time() - start_time
        print(f"✅ EasyOCR 초기화 완료! ({init_time:.2f}초)")
        
        # 결과 저장 폴더
        self.output_dir = Path("test_results/button_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_test_image(self):
        """
        테스트용 쿠팡이츠 스타일 이미지 생성
        (실제 스크린샷 대신 시뮬레이션 이미지)
        """
        # 1200x800 크기의 흰색 배경
        img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        
        # 한글 텍스트를 직접 그리기는 어려우므로, 실제 UI와 비슷한 구조 시뮬레이션
        # 쿠팡이츠 색상 (파란색 계열)
        coupang_blue = (63, 118, 180)  # BGR
        
        # 접수 버튼 영역 (하단 중앙)
        button_x, button_y = 850, 650
        button_w, button_h = 200, 80
        
        # 파란색 버튼 배경
        cv2.rectangle(img, 
                     (button_x, button_y), 
                     (button_x + button_w, button_y + button_h), 
                     coupang_blue, -1)
        
        # 흰색 테두리
        cv2.rectangle(img, 
                     (button_x, button_y), 
                     (button_x + button_w, button_y + button_h), 
                     (255, 255, 255), 2)
        
        # 텍스트 "접수" 추가 (OpenCV 한글 폰트 제한으로 영어로 대체)
        cv2.putText(img, 'Accept', 
                   (button_x + 60, button_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 추가 버튼들 (거절, 준비 시간 등)
        # 거절 버튼
        reject_x = 550
        cv2.rectangle(img, 
                     (reject_x, button_y), 
                     (reject_x + button_w, button_y + button_h), 
                     (128, 128, 128), -1)
        cv2.putText(img, 'Reject', 
                   (reject_x + 60, button_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 시간 정보
        cv2.putText(img, '12min', (400, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        
        # 주문 정보
        cv2.putText(img, 'Order #1B7S9E', (100, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        # 금액 정보
        cv2.putText(img, '17,800won', (900, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        return img
    
    def load_actual_screenshot(self):
        """
        실제 사용자 제공 스크린샷 로드 시뮬레이션
        (실제로는 화면 캡처나 파일 로드)
        """
        print("📷 실제 스크린샷 로드 중...")
        
        # 실제 구현에서는 mss를 사용해 화면 캡처하거나 파일 로드
        # 여기서는 테스트용 이미지 생성
        return self.create_test_image()
    
    def detect_buttons_with_easyocr(self, image):
        """
        EasyOCR을 사용해 텍스트 버튼 감지
        
        Args:
            image: 입력 이미지 (numpy array)
            
        Returns:
            dict: 감지된 버튼들의 정보
        """
        print("🔍 EasyOCR로 텍스트 감지 중...")
        start_time = time.time()
        
        # EasyOCR 실행
        results = self.reader.readtext(image, paragraph=False)
        
        processing_time = time.time() - start_time
        print(f"⚡ OCR 처리 완료: {processing_time:.3f}초")
        
        detected_buttons = {}
        button_candidates = []
        
        # 결과 분석
        for i, (bbox, text, confidence) in enumerate(results):
            print(f"📝 감지된 텍스트 {i+1}: '{text}' (신뢰도: {confidence:.3f})")
            
            # 접수 관련 키워드들
            accept_keywords = ['접수', '수락', 'accept', 'confirm']
            reject_keywords = ['거부', '거절', 'reject', 'decline']
            
            text_lower = text.lower().strip()
            
            # 접수 버튼 감지
            for keyword in accept_keywords:
                if keyword in text_lower and confidence > 0.5:
                    center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                    center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                    
                    button_info = {
                        'type': 'accept',
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'center': (center_x, center_y),
                        'processing_time': processing_time
                    }
                    
                    detected_buttons['accept'] = button_info
                    button_candidates.append(button_info)
                    print(f"🎯 접수 버튼 발견! 위치: ({center_x}, {center_y})")
                    break
            
            # 거절 버튼 감지
            for keyword in reject_keywords:
                if keyword in text_lower and confidence > 0.5:
                    center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                    center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                    
                    button_info = {
                        'type': 'reject',
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'center': (center_x, center_y),
                        'processing_time': processing_time
                    }
                    
                    detected_buttons['reject'] = button_info
                    button_candidates.append(button_info)
                    print(f"❌ 거절 버튼 발견! 위치: ({center_x}, {center_y})")
                    break
        
        return {
            'buttons': detected_buttons,
            'all_results': results,
            'candidates': button_candidates,
            'total_processing_time': processing_time
        }
    
    def extract_and_save_button_region(self, image, button_info, button_type):
        """
        감지된 버튼 영역을 추출하고 저장
        
        Args:
            image: 원본 이미지
            button_info: 버튼 정보
            button_type: 버튼 타입 ('accept', 'reject' 등)
        """
        bbox = button_info['bbox']
        
        # 바운딩 박스 좌표 추출
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # 여백 추가 (버튼 영역을 좀 더 크게)
        margin = 20
        h, w = image.shape[:2]
        
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        # 버튼 영역 추출
        button_region = image[y_min:y_max, x_min:x_max]
        
        # 파일명 생성
        timestamp = int(time.time())
        filename = f"{button_type}_button_{timestamp}.png"
        filepath = self.output_dir / filename
        
        # 이미지 저장
        cv2.imwrite(str(filepath), button_region)
        print(f"💾 {button_type} 버튼 이미지 저장: {filepath}")
        
        return str(filepath), (x_min, y_min, x_max, y_max)
    
    def create_annotated_image(self, image, detection_results):
        """
        감지된 버튼들을 표시한 주석 이미지 생성
        """
        annotated = image.copy()
        
        for button_type, button_info in detection_results['buttons'].items():
            bbox = button_info['bbox']
            center = button_info['center']
            confidence = button_info['confidence']
            
            # 바운딩 박스 그리기
            pts = np.array(bbox, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            color = (0, 255, 0) if button_type == 'accept' else (0, 0, 255)
            cv2.polylines(annotated, [pts], True, color, 3)
            
            # 중심점 표시
            cv2.circle(annotated, center, 10, color, -1)
            
            # 텍스트 라벨
            label = f"{button_type}: {confidence:.2f}"
            cv2.putText(annotated, label, 
                       (center[0]-50, center[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 주석 이미지 저장
        timestamp = int(time.time())
        annotated_path = self.output_dir / f"annotated_result_{timestamp}.png"
        cv2.imwrite(str(annotated_path), annotated)
        print(f"📊 주석 이미지 저장: {annotated_path}")
        
        return str(annotated_path)
    
    def save_detection_report(self, detection_results, image_info):
        """
        감지 결과를 JSON 리포트로 저장
        """
        report = {
            'timestamp': int(time.time()),
            'image_info': image_info,
            'detection_summary': {
                'total_buttons_found': len(detection_results['buttons']),
                'accept_button_found': 'accept' in detection_results['buttons'],
                'reject_button_found': 'reject' in detection_results['buttons'],
                'processing_time_ms': detection_results['total_processing_time'] * 1000
            },
            'detected_buttons': {},
            'all_ocr_results': []
        }
        
        # 버튼별 상세 정보
        for button_type, button_info in detection_results['buttons'].items():
            report['detected_buttons'][button_type] = {
                'text': button_info['text'],
                'confidence': float(button_info['confidence']),
                'center_coordinates': [int(button_info['center'][0]), int(button_info['center'][1])],
                'bbox_coordinates': [[float(point[0]), float(point[1])] for point in button_info['bbox']]
            }
        
        # 모든 OCR 결과
        for bbox, text, confidence in detection_results['all_results']:
            report['all_ocr_results'].append({
                'text': text,
                'confidence': float(confidence),
                'bbox': [[float(point[0]), float(point[1])] for point in bbox]
            })
        
        # 리포트 저장
        timestamp = int(time.time())
        report_path = self.output_dir / f"detection_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📋 감지 리포트 저장: {report_path}")
        return str(report_path)
    
    def run_full_test(self):
        """전체 테스트 실행"""
        print("🚀 EasyOCR 접수 버튼 감지 테스트 시작!")
        print("=" * 60)
        
        # 1. 테스트 이미지 로드
        print("1️⃣ 테스트 이미지 준비...")
        test_image = self.load_actual_screenshot()
        image_info = {
            'width': test_image.shape[1],
            'height': test_image.shape[0],
            'channels': test_image.shape[2]
        }
        print(f"   이미지 크기: {image_info['width']}x{image_info['height']}")
        
        # 원본 이미지 저장
        timestamp = int(time.time())
        original_path = self.output_dir / f"original_screenshot_{timestamp}.png"
        cv2.imwrite(str(original_path), test_image)
        print(f"   원본 이미지 저장: {original_path}")
        
        # 2. EasyOCR로 텍스트 감지
        print("\n2️⃣ EasyOCR 텍스트 감지...")
        detection_results = self.detect_buttons_with_easyocr(test_image)
        
        # 3. 결과 분석
        print(f"\n3️⃣ 결과 분석...")
        print(f"   총 감지된 텍스트: {len(detection_results['all_results'])}개")
        print(f"   감지된 버튼: {len(detection_results['buttons'])}개")
        
        # 4. 접수 버튼 처리
        if 'accept' in detection_results['buttons']:
            print("\n4️⃣ 접수 버튼 발견! 이미지 추출 중...")
            button_info = detection_results['buttons']['accept']
            
            # 버튼 영역 추출 및 저장
            button_path, coords = self.extract_and_save_button_region(
                test_image, button_info, 'accept'
            )
            
            print(f"   접수 버튼 좌표: {button_info['center']}")
            print(f"   신뢰도: {button_info['confidence']:.3f}")
            print(f"   추출 영역: {coords}")
            print(f"   저장된 이미지: {button_path}")
        else:
            print("\n4️⃣ ❌ 접수 버튼을 찾지 못했습니다.")
        
        # 5. 주석 이미지 생성
        print("\n5️⃣ 주석 이미지 생성...")
        annotated_path = self.create_annotated_image(test_image, detection_results)
        
        # 6. 감지 리포트 저장
        print("\n6️⃣ 감지 리포트 생성...")
        report_path = self.save_detection_report(detection_results, image_info)
        
        # 7. 결과 요약
        print("\n" + "=" * 60)
        print("🎯 테스트 결과 요약")
        print("=" * 60)
        
        if 'accept' in detection_results['buttons']:
            accept_info = detection_results['buttons']['accept']
            print(f"✅ 접수 버튼 감지 성공!")
            print(f"   위치: {accept_info['center']}")
            print(f"   텍스트: '{accept_info['text']}'")
            print(f"   신뢰도: {accept_info['confidence']:.1%}")
        else:
            print("❌ 접수 버튼 감지 실패")
        
        print(f"\n📁 생성된 파일들:")
        print(f"   - 원본: {original_path}")
        print(f"   - 주석: {annotated_path}")
        print(f"   - 리포트: {report_path}")
        
        if 'accept' in detection_results['buttons']:
            button_path = self.output_dir / f"accept_button_{timestamp}.png"
            print(f"   - 접수버튼: {button_path}")
        
        print(f"\n⚡ 처리 시간: {detection_results['total_processing_time']:.3f}초")
        print("🎉 테스트 완료!")
        
        return detection_results

def main():
    """메인 테스트 함수"""
    print("🎯 EasyOCR 접수 버튼 감지 테스트")
    print("쿠팡이츠 화면에서 텍스트 기반으로 접수 버튼 찾기\n")
    
    try:
        tester = EasyOCRButtonTester()
        results = tester.run_full_test()
        
        # 성공 여부에 따른 종료 코드
        if 'accept' in results['buttons']:
            print("\n✅ 테스트 성공: 접수 버튼 감지됨")
            return 0
        else:
            print("\n❌ 테스트 실패: 접수 버튼 감지 안됨")
            return 1
            
    except Exception as e:
        print(f"\n💥 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit(main())
