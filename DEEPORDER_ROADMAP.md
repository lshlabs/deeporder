
# 🚀 DeepOrder 프로젝트 상용화 로드맵 (0 to 100%)

이 문서는 배달앱 주문 처리 자동화 프로젝트인 **DeepOrder**를 현재의 개념 증명(PoC, 50~60%) 단계에서 상용화가 가능한 무결점 수준(100%)으로 끌어올리기 위한 단계별 청사진입니다.

## 진행도 업데이트 (2026-02-23)

- 전체 진행도(포트폴리오 완성 기준): **약 88~92%**
- 전체 진행도(상용화 기준): **약 70~75%**

### 이번 반영 완료 사항

* `utils/path_manager.py` 추가 (실행 위치/패키징 경로 대응)
* `dialog/*.py`의 `uic.loadUi('ui/...')` 상대 경로 제거 (경로 관리자 사용)
* `utils/data_manager.py` 경로 정규화 + 하위호환 기본값 처리(`settings`, `enabled`)
* `core_functions/vision_engine.py` 추가 (OCR/Template 전략 레이어)
* `core_functions/macro_runner.py` 개선
  * `start_macro(macro_key, run_options=None)` 지원
  * 타임아웃/재시도 옵션화
  * `stop_all()` 추가
  * 디버그 경로 하드코딩 제거
* `ui/MainWindow.ui` + `dialog/main_dialog.py` 로그 패널 연동
* 메인 창 포커스 기준 `F12` 긴급 중단 단축키 추가
* 실험/수동 검증 코드 분리
  * `experiments/`
  * `tests/manual/`
* 포트폴리오/검증/패키징 문서 추가
  * `docs/portfolio/BEFORE_AFTER.md`
  * `docs/portfolio/VALIDATION_SCENARIOS.md`
  * `docs/portfolio/PACKAGING_WINDOWS.md`
* Windows 패키징 준비물 추가
  * `DeepOrder.spec`
  * `scripts/build_windows.bat`
  * `main.py`

### 현재 남은 핵심 작업 (다음 진행)

* Windows 환경에서 `DeepOrder.spec` 실제 빌드 검증 (`dist/` 산출물 확인)
* (완료/보강 필요) GUI 로그/F12/타임아웃 캡처 확보 - 오프스크린 증빙 수집 완료, 실환경 캡처 추가 권장
* (부분 완료) 포트폴리오용 결과 증빙(로그 캡처, 실패/복구 사례) 추가
* (부분 완료) 배민/쿠팡 검증 로그 기록 - 샘플 스크린샷 기반 OCR 검증 완료, 실배달앱 실환경 검증 추가 필요
* (상용화 관점) 전역 핫키/오프라인 EasyOCR 모델 포함/운영 설정 UI 고도화

---

## Phase 1: 기반 구조 리팩토링 및 안정화 (현재 50% → 70%)
가장 시급한 과제는 실행 환경(디렉토리 위치, OS 등)에 의존하는 하드코딩된 경로를 제거하고, 향후 배포(.exe 파일 변환)를 대비한 절대 경로 체계를 구축하는 것입니다.

**상태:** `대부분 완료 (약 90%)`

### Action Items
* [x] 실행 위치와 무관하게 프로젝트 루트 디렉토리를 반환하는 경로 관리자 구현
* [x] `uic.loadUi`, `cv2.imread` 등에 사용되는 주요 경로(`ui/...`, `img/...`, `data.json`)를 경로 관리자 기반으로 치환
* [x] 실험용 코드(`test_*.py`)와 운영 코드 분리 (`experiments/`, `tests/manual/`)
* [ ] 잔여 실험 스크립트/문서 내 구 경로 표기 정리(선택)

### 핵심 코드 예시: 절대 경로 매니저 (`utils/path_manager.py`)
```python
import sys
from pathlib import Path

def get_base_dir() -> Path:
    """
    스크립트 모드와 PyInstaller 패키징 모드 모두에서 
    정확한 프로젝트 루트 경로를 반환합니다.
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller로 빌드된 실행 파일 환경
        return Path(sys._MEIPASS)
    else:
        # 일반 Python 스크립트 실행 환경 (이 파일의 부모의 부모 폴더)
        return Path(__file__).resolve().parent.parent

# 사용 예시 (UI 로드 또는 데이터 접근 시)
BASE_DIR = get_base_dir()
UI_FILE_PATH = BASE_DIR / "ui" / "MainWindow.ui"
DATA_FILE_PATH = BASE_DIR / "utils" / "data.json"

```

---

## Phase 2: 하이브리드 비전 엔진 통합 (70% → 85%)

현재 개별적으로 동작하는 OpenCV 템플릿 매처와 EasyOCR 텍스트 매처를 하나의 '전략 패턴(Strategy Pattern)'으로 통합합니다. 사용자가 매크로 단계별로 적절한 방식을 선택할 수 있게 합니다.

**상태:** `부분 완료 (약 60~70%)`

### Action Items

* [x] 공통 인터페이스 성격의 전략 레이어(`core_functions/vision_engine.py`) 추가
* [x] `MacroRunner`가 구체 OCR 매처 직접 참조 대신 `VisionEngine` 사용
* [~] EasyOCR ROI 최적화: 기존 `image_matcher_easyocr.py`의 앱별 ROI/키워드 로직 유지(추가 정교화는 잔여)
* [~] 해상도/DPI 보정 로직: 기존 템플릿 스케일 좌표 계산 유지, 실환경 검증 추가 필요
* [ ] 단계별(UI) 매처 선택 기능 노출

### 핵심 코드 예시: 매칭 전략 패턴 (`core_functions/vision_engine.py`)

```python
from abc import ABC, abstractmethod
import cv2
import easyocr

class BaseMatcher(ABC):
    @abstractmethod
    def find_target(self, screen_image, target_data):
        pass

class TemplateMatcher(BaseMatcher):
    def find_target(self, screen_image, template_path):
        template = cv2.imread(template_path)
        # OpenCV 템플릿 매칭 로직
        result = cv2.matchTemplate(screen_image, template, cv2.TM_CCOEFF_NORMED)
        # 최적 좌표 계산 및 반환 (생략)
        return target_x, target_y

class OCRMatcher(BaseMatcher):
    def __init__(self):
        # 메모리 효율을 위해 싱글톤으로 로드 권장
        self.reader = easyocr.Reader(['ko', 'en'])

    def find_target(self, screen_image, keyword):
        results = self.reader.readtext(screen_image)
        for bbox, text, prob in results:
            if keyword in text.replace(" ", ""):
                # Bounding Box 중앙값 계산 (생략)
                return center_x, center_y
        return None

# 매크로 실행부에서의 활용
def execute_step(screen, step_data):
    matcher = OCRMatcher() if step_data['method'] == 'ocr' else TemplateMatcher()
    target_pos = matcher.find_target(screen, step_data['target'])
    return target_pos

```

---

## Phase 3: UX/UI 강화 및 예외 처리 (85% → 95%)

콘솔 창(Terminal) 없이도 사용자가 현재 매크로의 진행 상태를 명확히 알 수 있도록 GUI 로그 패널을 연동하고, 무한 대기를 방지하는 타임아웃 방어 로직을 세웁니다.

**상태:** `대부분 완료 (약 80~90%)`

### Action Items

* [x] 파이썬 `logging` 형식 로그를 메인 GUI 로그 패널(`QPlainTextEdit`)에 표시
* [x] 화면 탐색 타임아웃/재시도 로직 및 옵션화 구현 (`MacroRunner`)
* [x] 긴급 중단 단축키(F12) 연동 (메인 창 포커스 기준)
* [ ] 전역(시스템) 핫키 지원 (상용화 단계)
* [ ] 실제 사용자 시나리오 기반 로그/에러 메시지 다듬기

### 핵심 코드 예시: GUI 실시간 로그 연동 및 타임아웃 (`utils/logger_ui.py`)

```python
import logging
import time
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QTextBrowser

class QLogSignal(QObject):
    update_log = pyqtSignal(str)

class GUILogHandler(logging.Handler):
    def __init__(self, text_widget: QTextBrowser):
        super().__init__()
        self.text_widget = text_widget
        self.signals = QLogSignal()
        self.signals.update_log.connect(self.text_widget.append)

    def emit(self, record):
        msg = self.format(record)
        self.signals.update_log.emit(msg)

# 타임아웃 방어 로직 예시
def wait_for_element(matcher, screen_monitor, target, timeout_sec=10):
    start_time = time.time()
    while (time.time() - start_time) < timeout_sec:
        screen = screen_monitor.capture()
        pos = matcher.find_target(screen, target)
        if pos:
            return pos
        time.sleep(0.5) # CPU 과점유 방지
    
    logging.error(f"[{target}] 요소를 {timeout_sec}초 내에 찾지 못했습니다.")
    raise TimeoutError("Element search timeout")

```

---

## Phase 4: 배포 및 상용화 준비 (95% → 100%)

비개발자도 파이썬 설치 없이 원클릭으로 실행할 수 있도록 패키징하고 최종 사용 문서를 작성합니다.

**상태:** `준비 완료 단계 (약 50~60%)`

### Action Items

* [ ] `PyInstaller`를 활용한 단일 실행 파일(.exe / .app) 실제 빌드 검증
* [x] EasyOCR/UI/이미지 에셋 포함을 고려한 `spec` 파일 작성 (`DeepOrder.spec`)
* [x] README.md 개편 (실행/구조/포트폴리오 문서 링크 포함)
* [x] Windows 빌드 스크립트 및 가이드 문서 작성 (`scripts/build_windows.bat`, `docs/portfolio/PACKAGING_WINDOWS.md`)
* [ ] EasyOCR 모델 포함/오프라인 배포 검증

---

## 다음 진행 우선순위 (실행 순서)

### 1) 실환경 검증 증빙 확보 (최우선)
* [x] `main.py` UI 기준 GUI 로그 패널/`F12`/타임아웃 캡처 확보 (오프스크린 자동 수집)
* [x] 배민/쿠팡 샘플 스크린샷 기반 OCR 검증 로그/스크린샷 수집
* [ ] 실배달앱 환경에서 배민/쿠팡 각각 최소 1회 시도 로그 + 결과 스크린샷 확보
* `docs/portfolio/VALIDATION_SCENARIOS.md`에 실제 결과(성공/실패/원인) 추가

### 2) Windows 빌드 검증 (별도 Windows 환경)
* `scripts\\build_windows.bat` 실행
* `dist/` 산출물 실행 확인
* 누락 모듈/에셋 발생 시 `DeepOrder.spec` 보정 후 재기록

### 3) 상용화 격차 축소 (후속)
* 전역 핫키
* 운영 설정 UI(매처 모드/재시도/타임아웃) 노출
* EasyOCR 모델 번들링/초기 다운로드 UX 개선

### 핵심 적용 예시: PyInstaller 빌드 명령어 (`build.sh` 또는 터미널)

```bash
# Pyinstaller를 통한 단일 실행 파일 생성 명령어 예시
# --hidden-import로 EasyOCR 의존성 누락 방지, --add-data로 에셋 포함
pyinstaller --name "DeepOrder" \
            --windowed \
            --noconfirm \
            --icon=img/app_icon.ico \
            --add-data "ui/*;ui" \
            --add-data "img/*;img" \
            --add-data "utils/data.json;utils" \
            --hidden-import "easyocr" \
            --hidden-import "cv2" \
            dialog/main_dialog.py

```
