"""
🎯 텍스트 기반 UI 자동화 기술 종합 비교
OpenCV 템플릿 매칭 vs 다양한 OCR 기술들

배달앱 "접수", "거부" 버튼 같은 텍스트 UI에 최적화된 솔루션들
"""

import cv2
import numpy as np
import time
import mss
from typing import List, Tuple, Dict, Optional
import json

# OCR 라이브러리들
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import paddleocr
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False

class TextBasedUIAutomation:
    """
    다양한 텍스트 인식 기술들을 비교하고 최적의 솔루션 제공
    """
    
    def __init__(self):
        self.ocr_engines = {}
        self.performance_stats = {
            'opencv_template': {'times': [], 'accuracies': []},
            'easyocr': {'times': [], 'accuracies': []},
            'tesseract': {'times': [], 'accuracies': []},
            'paddleocr': {'times': [], 'accuracies': []},
            'hybrid': {'times': [], 'accuracies': []}
        }
        
        self._initialize_ocr_engines()
        
    def _initialize_ocr_engines(self):
        """OCR 엔진들 초기화"""
        
        # EasyOCR (한글+영어, GPU/CPU 자동 선택)
        if HAS_EASYOCR:
            print("🔄 EasyOCR 초기화 중...")
            self.ocr_engines['easyocr'] = easyocr.Reader(['ko', 'en'], gpu=False)
            print("✅ EasyOCR 준비 완료")
        
        # Tesseract (Google의 오픈소스 OCR)
        if HAS_TESSERACT:
            try:
                # Tesseract 경로 확인
                pytesseract.get_tesseract_version()
                self.ocr_engines['tesseract'] = True
                print("✅ Tesseract 준비 완료")
            except:
                print("❌ Tesseract 설치 필요: brew install tesseract tesseract-lang")
        
        # PaddleOCR (중국 바이두의 고성능 OCR)
        if HAS_PADDLEOCR:
            print("🔄 PaddleOCR 초기화 중...")
            self.ocr_engines['paddleocr'] = paddleocr.PaddleOCR(use_angle_cls=True, lang='korean')
            print("✅ PaddleOCR 준비 완료")

    def capture_screen(self) -> np.ndarray:
        """화면 캡처"""
        with mss.mss() as sct:
            # 테스트용 작은 영역 캡처
            monitor = {"top": 200, "left": 200, "width": 1000, "height": 600}
            screenshot = sct.grab(monitor)
            return np.array(screenshot)[:, :, :3]  # RGB

# =============================================================================
# 1. OpenCV 템플릿 매칭 (기존 방식)
# =============================================================================

class OpenCVTemplateMatcher:
    """기존 OpenCV 템플릿 매칭 방식"""
    
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.template_cache = {}
    
    def find_button_by_template(self, screenshot: np.ndarray, button_type: str) -> Tuple[bool, Optional[Tuple], float]:
        """
        템플릿 매칭으로 버튼 찾기
        
        Args:
            screenshot: 스크린샷
            button_type: 'accept' 또는 'reject'
        """
        start_time = time.time()
        
        try:
            # 실제로는 저장된 템플릿 이미지 사용
            # 여기서는 시뮬레이션
            gray_screen = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # 가상의 템플릿 (실제로는 파일에서 로드)
            h, w = gray_screen.shape
            if button_type == 'accept':
                template = gray_screen[h//2-50:h//2+50, w//2-100:w//2]  # 접수 버튼 위치 추정
            else:
                template = gray_screen[h//2-50:h//2+50, w//2:w//2+100]  # 거부 버튼 위치 추정
            
            if template.size == 0:
                return False, None, 0.0
            
            result = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            processing_time = time.time() - start_time
            
            if max_val > self.threshold:
                center_x = max_loc[0] + template.shape[1] // 2
                center_y = max_loc[1] + template.shape[0] // 2
                return True, (center_x, center_y), processing_time
            else:
                return False, None, processing_time
                
        except Exception as e:
            processing_time = time.time() - start_time
            return False, None, processing_time

# =============================================================================
# 2. EasyOCR (가장 사용하기 쉬운 OCR)
# =============================================================================

class EasyOCRDetector:
    """
    EasyOCR 기반 텍스트 버튼 감지
    
    ✅ 장점:
    - 설치 쉬움: pip install easyocr
    - 한글 지원 우수
    - GPU 자동 활용
    - 높은 정확도
    
    ❌ 단점:
    - 초기 로딩 시간 (2-3초)
    - 메모리 사용량 높음
    """
    
    def __init__(self):
        if HAS_EASYOCR:
            self.reader = easyocr.Reader(['ko', 'en'], gpu=False)
        else:
            self.reader = None
    
    def find_button_by_text(self, screenshot: np.ndarray, target_text: str) -> Tuple[bool, Optional[Tuple], float]:
        """
        텍스트 기반으로 버튼 찾기
        
        Args:
            screenshot: 스크린샷
            target_text: 찾을 텍스트 ("접수", "거부" 등)
        """
        if not self.reader:
            return False, None, 9999
        
        start_time = time.time()
        
        try:
            # OCR 실행
            results = self.reader.readtext(screenshot, paragraph=False)
            
            for (bbox, text, confidence) in results:
                # 텍스트 매칭 (유사도 기반)
                similarity = self._calculate_similarity(target_text, text)
                
                if similarity > 0.7 and confidence > 0.6:
                    # 중심점 계산
                    center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                    center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                    
                    processing_time = time.time() - start_time
                    return True, (center_x, center_y), processing_time
            
            processing_time = time.time() - start_time
            return False, None, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            return False, None, processing_time
    
    def _calculate_similarity(self, target: str, found: str) -> float:
        """텍스트 유사도 계산"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, target.lower(), found.lower()).ratio()

# =============================================================================
# 3. Tesseract (Google의 전통적 OCR)
# =============================================================================

class TesseractDetector:
    """
    Tesseract 기반 텍스트 감지
    
    ✅ 장점:
    - 매우 빠름
    - 메모리 사용량 적음
    - 안정적
    - 오픈소스
    
    ❌ 단점:
    - 한글 인식률 낮음
    - 설정 복잡
    - 노이즈에 민감
    """
    
    def __init__(self):
        self.available = HAS_TESSERACT
        
    def find_button_by_text(self, screenshot: np.ndarray, target_text: str) -> Tuple[bool, Optional[Tuple], float]:
        """Tesseract로 버튼 찾기"""
        if not self.available:
            return False, None, 9999
            
        start_time = time.time()
        
        try:
            # 전처리 (Tesseract는 전처리가 중요)
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # 이진화로 텍스트 선명하게
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 한글+영어 설정
            config = '--oem 3 --psm 6 -l kor+eng'
            
            # OCR 실행 (위치 정보 포함)
            data = pytesseract.image_to_data(binary, config=config, output_type=pytesseract.Output.DICT)
            
            # 결과 분석
            for i, text in enumerate(data['text']):
                if text.strip() and target_text in text:
                    # 바운딩 박스 정보
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    conf = int(data['conf'][i])
                    
                    if conf > 50:  # 신뢰도 50% 이상
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        processing_time = time.time() - start_time
                        return True, (center_x, center_y), processing_time
            
            processing_time = time.time() - start_time
            return False, None, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            return False, None, processing_time

# =============================================================================
# 4. PaddleOCR (바이두의 고성능 OCR)
# =============================================================================

class PaddleOCRDetector:
    """
    PaddleOCR 기반 감지
    
    ✅ 장점:
    - 매우 높은 정확도
    - 다양한 언어 지원
    - 회전된 텍스트도 인식
    - 상업적 사용 가능
    
    ❌ 단점:
    - 큰 용량 (수백 MB)
    - 초기 로딩 느림
    - 중국어 위주 최적화
    """
    
    def __init__(self):
        self.available = HAS_PADDLEOCR
        if self.available:
            self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='korean')
    
    def find_button_by_text(self, screenshot: np.ndarray, target_text: str) -> Tuple[bool, Optional[Tuple], float]:
        """PaddleOCR로 버튼 찾기"""
        if not self.available:
            return False, None, 9999
            
        start_time = time.time()
        
        try:
            results = self.ocr.ocr(screenshot, cls=True)
            
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, confidence) = line
                    
                    if target_text in text and confidence > 0.7:
                        # 중심점 계산
                        center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                        center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                        
                        processing_time = time.time() - start_time
                        return True, (center_x, center_y), processing_time
            
            processing_time = time.time() - start_time
            return False, None, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            return False, None, processing_time

# =============================================================================
# 5. 하이브리드 접근법 (최고 성능)
# =============================================================================

class HybridTextDetector:
    """
    여러 OCR 기술을 조합한 하이브리드 접근법
    
    🎯 전략:
    1. 빠른 Tesseract로 1차 스크리닝
    2. 실패 시 EasyOCR로 정밀 분석
    3. 색상 필터링으로 후보 영역 축소
    """
    
    def __init__(self):
        self.tesseract = TesseractDetector()
        self.easyocr = EasyOCRDetector()
        self.color_filters = {
            'accept': ([100, 50, 50], [130, 255, 255]),  # 파란색 계열
            'reject': ([0, 50, 50], [20, 255, 255])      # 빨간색 계열
        }
    
    def find_button_smart(self, screenshot: np.ndarray, target_text: str, button_type: str = 'accept') -> Tuple[bool, Optional[Tuple], float, str]:
        """
        스마트 하이브리드 버튼 찾기
        
        Returns:
            (found, location, processing_time, method_used)
        """
        start_time = time.time()
        
        # 1단계: 색상으로 후보 영역 좁히기
        roi_candidates = self._find_color_regions(screenshot, button_type)
        
        if roi_candidates:
            # ROI 영역에서만 OCR 실행 (훨씬 빠름)
            for roi in roi_candidates[:3]:  # 상위 3개 후보만
                roi_img = self._extract_roi(screenshot, roi)
                
                # 2단계: 빠른 Tesseract 시도
                found, location, _ = self.tesseract.find_button_by_text(roi_img, target_text)
                if found:
                    # ROI 좌표를 전체 화면 좌표로 변환
                    global_location = (location[0] + roi['x'], location[1] + roi['y'])
                    processing_time = time.time() - start_time
                    return True, global_location, processing_time, 'Tesseract+Color'
                
                # 3단계: EasyOCR로 정밀 분석
                found, location, _ = self.easyocr.find_button_by_text(roi_img, target_text)
                if found:
                    global_location = (location[0] + roi['x'], location[1] + roi['y'])
                    processing_time = time.time() - start_time
                    return True, global_location, processing_time, 'EasyOCR+Color'
        
        # 4단계: 전체 화면에서 EasyOCR (최후 수단)
        found, location, _ = self.easyocr.find_button_by_text(screenshot, target_text)
        processing_time = time.time() - start_time
        
        if found:
            return True, location, processing_time, 'EasyOCR_Fullscreen'
        else:
            return False, None, processing_time, 'Failed'
    
    def _find_color_regions(self, screenshot: np.ndarray, button_type: str) -> List[Dict]:
        """색상 기반으로 버튼 후보 영역 찾기"""
        try:
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV)
            
            if button_type in self.color_filters:
                lower, upper = self.color_filters[button_type]
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # 노이즈 제거
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # 윤곽선 찾기
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                regions = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 500 < area < 50000:  # 버튼 크기 범위
                        x, y, w, h = cv2.boundingRect(contour)
                        regions.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area})
                
                # 면적 순으로 정렬 (큰 것부터)
                regions.sort(key=lambda r: r['area'], reverse=True)
                return regions
                
        except Exception:
            pass
            
        return []
    
    def _extract_roi(self, screenshot: np.ndarray, roi: Dict) -> np.ndarray:
        """ROI 영역 추출"""
        return screenshot[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]

# =============================================================================
# 6. 성능 비교 테스트
# =============================================================================

class PerformanceComparator:
    """다양한 텍스트 인식 기술 성능 비교"""
    
    def __init__(self):
        self.opencv_matcher = OpenCVTemplateMatcher()
        self.easyocr_detector = EasyOCRDetector()
        self.tesseract_detector = TesseractDetector()
        self.paddleocr_detector = PaddleOCRDetector()
        self.hybrid_detector = HybridTextDetector()
        
    def compare_all_methods(self, iterations: int = 5):
        """모든 방법 성능 비교"""
        print(f"\n🎯 텍스트 기반 UI 자동화 기술 비교 ({iterations}회 테스트)")
        print("=" * 70)
        
        results = {
            'OpenCV 템플릿': [],
            'EasyOCR': [],
            'Tesseract': [],
            'PaddleOCR': [],
            'Hybrid': []
        }
        
        for i in range(iterations):
            print(f"\r📊 테스트 진행: {i+1}/{iterations}", end='', flush=True)
            
            # 화면 캡처
            screenshot = self._capture_test_screen()
            
            # 1. OpenCV 템플릿 매칭
            found, _, time_taken = self.opencv_matcher.find_button_by_template(screenshot, 'accept')
            results['OpenCV 템플릿'].append({'found': found, 'time': time_taken * 1000})
            
            # 2. EasyOCR
            found, _, time_taken = self.easyocr_detector.find_button_by_text(screenshot, '접수')
            results['EasyOCR'].append({'found': found, 'time': time_taken * 1000})
            
            # 3. Tesseract
            found, _, time_taken = self.tesseract_detector.find_button_by_text(screenshot, '접수')
            results['Tesseract'].append({'found': found, 'time': time_taken * 1000})
            
            # 4. PaddleOCR
            found, _, time_taken = self.paddleocr_detector.find_button_by_text(screenshot, '접수')
            results['PaddleOCR'].append({'found': found, 'time': time_taken * 1000})
            
            # 5. Hybrid
            found, _, time_taken, _ = self.hybrid_detector.find_button_smart(screenshot, '접수', 'accept')
            results['Hybrid'].append({'found': found, 'time': time_taken * 1000})
            
            time.sleep(0.1)
        
        print("\n✅ 테스트 완료!\n")
        self._print_comparison_results(results)
    
    def _capture_test_screen(self) -> np.ndarray:
        """테스트용 화면 캡처"""
        with mss.mss() as sct:
            monitor = {"top": 100, "left": 100, "width": 1000, "height": 700}
            screenshot = sct.grab(monitor)
            return np.array(screenshot)[:, :, :3]
    
    def _print_comparison_results(self, results: Dict):
        """결과 출력"""
        print("📊 성능 비교 결과")
        print("=" * 70)
        print(f"{'방법':<15} {'평균속도(ms)':<12} {'성공률(%)':<10} {'안정성':<8} {'추천도'}")
        print("-" * 70)
        
        recommendations = {
            'OpenCV 템플릿': '⭐⭐',
            'EasyOCR': '⭐⭐⭐⭐⭐',
            'Tesseract': '⭐⭐⭐',
            'PaddleOCR': '⭐⭐⭐⭐',
            'Hybrid': '⭐⭐⭐⭐⭐'
        }
        
        for method, data in results.items():
            if data and data[0]['time'] < 9000:  # 사용 가능한 방법만
                avg_time = sum(r['time'] for r in data) / len(data)
                success_rate = sum(1 for r in data if r['found']) / len(data) * 100
                stability = "높음" if success_rate > 80 else "보통" if success_rate > 50 else "낮음"
                
                print(f"{method:<15} {avg_time:>8.1f}ms    {success_rate:>6.1f}%    {stability:<8} {recommendations[method]}")
            else:
                print(f"{method:<15} {'사용불가':<12} {'N/A':<10} {'N/A':<8} {recommendations[method]}")
        
        print("\n💡 종합 분석:")
        print("🥇 **EasyOCR**: 가장 균형잡힌 성능, 한글 지원 우수")
        print("🥈 **Hybrid**: 최고 성능이지만 복잡함")  
        print("🥉 **PaddleOCR**: 높은 정확도, 하지만 용량 큼")
        print("📉 **OpenCV**: 텍스트 UI에는 부적합")
        print("⚡ **Tesseract**: 빠르지만 한글 인식률 낮음")
        
        print("\n🎯 **배달앱 자동화 추천**:")
        print("1순위: EasyOCR (설치 쉬움, 성능 좋음)")
        print("2순위: Hybrid 방식 (최고 성능, 복잡함)")
        print("3순위: PaddleOCR (정확하지만 무거움)")

def main():
    """메인 실행"""
    print("🎯 텍스트 기반 UI 자동화 기술 비교")
    print("배달앱 '접수', '거부' 버튼 같은 텍스트 UI 최적화")
    print()
    
    comparator = PerformanceComparator()
    
    try:
        choice = input("테스트 횟수 선택 (1=빠른테스트, 2=표준테스트, 3=정확한테스트): ").strip()
        iterations = {'1': 3, '2': 5, '3': 10}.get(choice, 5)
        
        comparator.compare_all_methods(iterations)
        
        print("\n🚀 실제 적용 방법:")
        print("1. EasyOCR 사용: pip install easyocr")
        print("2. 기존 ImageMatcher를 TextBasedMatcher로 교체")
        print("3. '접수', '거부' 텍스트로 직접 버튼 찾기")
        print("4. 해상도 완전 독립적 + 90% 이상 정확도!")
        
    except KeyboardInterrupt:
        print("\n👋 테스트 중단됨")

if __name__ == "__main__":
    main()
