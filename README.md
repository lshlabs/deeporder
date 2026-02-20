# DeepOrder

PyQt6 기반 GUI + 화면 인식(OpenCV/EasyOCR) + 자동 클릭(pyautogui)로 구성된  
배달앱 주문 처리 자동화 실험 프로젝트입니다.

이 저장소는 **완료형 제품보다는 기능 검증/개선 중인 개발 단계 프로젝트**에 가깝습니다.

## 현재 상태

- 메인 GUI는 동작 가능: `dialog/main_dialog.py`
- 매크로 데이터는 JSON 파일(`utils/data.json`)에 저장
- 실행 엔진은 `MacroRunner` + `ScreenMonitor` + `MouseController`
- 이미지 매칭은 기존 OpenCV 매처(`core_functions/image_matcher.py`)와  
  EasyOCR 기반 매처(`image_matcher_easyocr.py`)가 공존
- 성능/정확도 실험 스크립트 다수 포함 (`ocr_test.py`, `test_*`, `performance_demo.py`)

## 기술 스택

- Python 3.x
- PyQt6
- OpenCV (`opencv-python`)
- EasyOCR
- mss
- pyautogui
- numpy
- Pillow

## 디렉터리 구조

```text
deeporder/
├─ dialog/                    # 메인 UI/다이얼로그 로직
├─ ui/                        # Qt Designer .ui 및 변환 파일
├─ core_functions/            # 매크로 실행, 화면 모니터링, 클릭 제어
├─ utils/                     # DataManager/TempManager/설정 데이터
├─ image_matcher_easyocr.py   # EasyOCR 기반 배달앱 특화 매처
├─ ocr_test.py                # OCR 기반 모니터링 실험 스크립트
├─ test_*.py                  # 기능 검증용 테스트 스크립트
├─ img/                       # 템플릿/디버깅 이미지
└─ test_results/              # 테스트 결과 이미지
```

## 빠른 실행

### 1) 가상환경 준비 (예시)

```bash
cd /Users/mac/Documents/GitHub/myGit/deeporder
python3 -m venv .venv
source .venv/bin/activate
```

### 2) 의존성 설치

```bash
pip install pyqt6 opencv-python numpy mss pyautogui pillow easyocr
```

### 3) 메인 GUI 실행

```bash
python3 dialog/main_dialog.py
```

### 4) OCR 모니터링 실험 실행(선택)

```bash
python3 ocr_test.py
```

## 핵심 모듈

- `dialog/main_dialog.py`  
  메인 화면, 매크로 목록 관리, 실행/중지 UI 이벤트 처리

- `core_functions/macro_runner.py`  
  매크로별 스레드 실행, 템플릿 탐색 재시도, 액션 클릭 실행, 디버깅 이미지 저장

- `core_functions/image_matcher.py`  
  템플릿-액션 데이터 로드, 화면 캡처, 템플릿 매칭, 스케일 좌표 변환

- `image_matcher_easyocr.py`  
  EasyOCR 싱글톤 리더 재사용, 배달앱별 ROI/키워드 기반 버튼 탐지

- `utils/data_manager.py`  
  매크로/설정 JSON 저장, 위저드 액션 생성, 매크로 복제와 이미지 경로 갱신

## 알려진 제약

- 경로 의존 코드가 일부 존재하여 실행 위치에 따라 오류가 날 수 있음  
  (예: `uic.loadUi('ui/MainWindow.ui', self)`)
- 로그 UI가 아직 미완성이라 일부 로그는 콘솔 출력에 의존
- 테스트/실험 스크립트와 운영 코드가 함께 있어 구조 정리가 더 필요
- 플랫폼/해상도 별 마우스 제어 안정성 추가 검증 필요

## 다음 정리 우선순위 제안

1. 경로/설정 관리 통합 (하드코딩 경로 제거)
2. 실행 로그 패널 UI 추가
3. EasyOCR/Template matcher 선택 전략 통일
4. 테스트 스크립트와 운영 코드 분리
5. requirements 파일 고정 및 실행 가이드 표준화

