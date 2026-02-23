# DeepOrder

PyQt6 기반 GUI + 화면 인식(OpenCV/EasyOCR) + 자동 클릭(pyautogui)로 구성된 배달앱 주문 처리 자동화 프로젝트입니다.

이번 정리에서는 PoC 성격의 코드를 포트폴리오 설명에 유리한 형태로 리팩토링했습니다.

## 이번 개선 핵심

- 경로 관리 통합: `utils/path_manager.py` (실행 위치/패키징 고려)
- GUI 로그 패널 추가: 메인 화면에서 실행 로그 확인 가능
- `MacroRunner` 안정화: 타임아웃/재시도 옵션 + 정리 로직 강화
- 비전 엔진 추상화: `core_functions/vision_engine.py` (OCR/Template 전략 경계)
- 구조 정리: 실험 스크립트 `experiments/`, 수동 테스트 `tests/manual/`
- Windows 패키징 준비물 추가: `DeepOrder.spec`, `scripts/build_windows.bat`

## 빠른 실행

### 1) 가상환경 준비

```bash
cd /Users/mac/Documents/GitHub/myGit/deeporder
python3 -m venv .venv
source .venv/bin/activate
```

### 2) 의존성 설치

```bash
pip install -r requirements.txt
```

### 3) 실행

```bash
python3 main.py
```

## 디렉터리 구조 (요약)

```text
deeporder/
├─ main.py                    # 앱 엔트리포인트 (패키징용)
├─ dialog/                    # 메인 UI/다이얼로그 로직
├─ ui/                        # Qt Designer .ui 파일
├─ core_functions/            # 실행 엔진, 마우스 제어, 비전 엔진
├─ utils/                     # 데이터/임시/경로/로그 유틸리티
├─ experiments/               # 성능 비교/실험 스크립트
├─ tests/manual/              # 수동 검증용 스크립트
├─ docs/portfolio/            # 포트폴리오 설명 문서
├─ img/                       # 템플릿/디버그 이미지
└─ DeepOrder.spec             # Windows 패키징 준비 파일
```

## 핵심 모듈

- `dialog/main_dialog.py`: 메인 화면, 매크로 리스트/실행/중지, GUI 로그 출력
- `core_functions/macro_runner.py`: 매크로 스레드 실행, 타임아웃/재시도, 디버그 저장
- `core_functions/vision_engine.py`: OCR/Template matcher 추상화 레이어
- `core_functions/image_matcher.py`: 템플릿 매칭 + 액션 좌표 계산
- `image_matcher_easyocr.py`: EasyOCR 기반 배달앱 버튼 탐지
- `utils/data_manager.py`: 매크로 데이터 로드/저장, 하위호환 경로 정규화
- `utils/path_manager.py`: 리소스 경로/패키징 경로 통합

## 포트폴리오 문서

- `docs/portfolio/BEFORE_AFTER.md`
- `docs/portfolio/VALIDATION_SCENARIOS.md`
- `docs/portfolio/PACKAGING_WINDOWS.md`

## 수동 검증 스크립트 예시

```bash
python3 -m tests.manual.test_easyocr_button_detection
python3 -m tests.manual.test_roi_limited_detection
```

## 실험 스크립트 예시

```bash
python3 -m experiments.ocr_test
python3 -m experiments.performance_demo
```

## 알려진 한계

- 실제 배달앱 환경에서 대규모 실사용 QA는 아직 별도 검증 필요
- EasyOCR 오프라인 배포(모델 포함) 최적화는 후속 작업 필요
- 전역 시스템 단축키는 미구현 (현재는 메인 창 포커스 기준 `F12`)
