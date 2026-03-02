# DeepOrder 전체 코드베이스 심층 리서치 보고서

작성일: 2026-03-02  
분석 대상 루트: `/Users/mac/Documents/GitHub/myGit/deeporder`

## 1. 분석 범위와 방법

이 보고서는 저장소의 실행 코드, UI 정의, 설정/데이터, 수동 테스트, 실험 코드, 배포 스크립트, 포트폴리오 문서를 모두 읽고 상호 참조한 결과다.

- 코드 파일
  - Python: 61개 (`runtime 28`, `experiments 8`, `manual tests 5`, `example/example_code 7`, 기타 포함)
  - UI XML: 10개 (`ui/*.ui`)
  - UI 생성 파이썬: 10개 (`ui/*_ui.py`)
- 문서/설정
  - Markdown: 8개
  - JSON: 4개 (실사용/증빙)
  - Spec/스크립트: `DeepOrder.spec`, `scripts/*.bat`, `scripts/*.ps1`, `scripts/*.py`
- 자산
  - PNG: 129개 (템플릿/디버그/테스트 결과)

추가 검증:
- 소스 문법 확인: `python3 -m py_compile ...` (프로젝트 소스 대상) 통과
- 구조적 grep: 클래스/함수/스텁/하드코딩 경로/`sys.exit` 패턴 점검

## 2. 프로젝트 한 줄 요약

DeepOrder는 **PyQt6 GUI 기반 매크로 편집기 + 화면 인식(OpenCV/EasyOCR) + 자동 클릭(pyautogui)**를 결합한 배달앱 주문 처리 자동화 도구이며, 현재는 **"운영 경로"와 "실험 경로"가 혼재한 상태의 포트폴리오형 PoC+** 단계다.

## 3. 최상위 아키텍처

### 3.1 실행 진입점
- `main.py`는 `dialog.main_dialog.main()`만 호출한다.
- 실질 앱 부트스트랩은 `dialog/main_dialog.py`에 집중되어 있다.

### 3.2 런타임 계층(핵심)
- UI 계층: `dialog/*`
  - 메인 창, 매크로 편집, 위저드(step1/2/3), 딜레이/설정/미리보기
- 실행 엔진: `core_functions/*`
  - `MacroRunner`: 스레드 실행/중지/재시도/타임아웃/로그
  - `MouseController`: 좌표 클릭
  - `VisionEngine`: 템플릿/OCR 매처 파사드
  - `ImageMatcher`: OpenCV 템플릿 매칭 + 액션 좌표 계산
- 데이터/유틸: `utils/*`
  - `DataManager`: `utils/data.json`의 매크로 CRUD/정규화
  - `TempManager`: 위저드 중간 상태 저장(`temp/tempdata.json`)
  - `path_manager`: 경로 추상화(PyInstaller 대응)
  - `logger_ui`: GUI 로그 핸들러

### 3.3 실험/검증 계층
- `experiments/*`: OCR/템플릿/하이브리드 성능 실험 및 데모
- `tests/manual/*`: 수동 실행 테스트 스크립트
- `scripts/collect_validation_evidence.py`: 오프스크린 GUI+샘플 이미지 증빙 수집

## 4. 실제 동작 플로우 (런타임 기준)

### 4.1 매크로 생성
1. 메인 창 `추가` 클릭 → `ActionWizardDialog`
2. Step1: 원본 스크린샷 선택
3. Step2/3: 드래그로 버튼 영역 지정(`plus/minus/time/reject/accept`)
4. `DataManager.create_wizard_actions()`가 A1~A6 액션 생성, 이미지 복사, 좌표 저장

### 4.2 매크로 실행
1. 메인 창 `RUN` 라벨 클릭
2. `MacroRunner.start_macro(macro_key)`
3. `macro settings`에서 재시도/간격/타임아웃/매처 모드 병합
4. 스레드 `_run_macro()` 시작
5. `원본 이미지(A1)`를 템플릿 ID(`M*_A1`)로 탐색
6. 찾으면 액션 좌표 계산 후 `MouseController.click_all_actions()` 순차 클릭
7. 성공/실패 로그 emit, 필요 시 디버그 이미지 저장
8. 종료 시 상태 정리 + UI 상태 복귀

### 4.3 긴급 중단
- 메인 윈도우 포커스 상태에서 `F12` → `stop_all_running_macros()`
- `MacroRunner.stop_all()` 호출 + GUI 로그 기록

## 5. 데이터 모델 (`utils/data.json`)

현재 데이터 구조:
- `macro_list`
  - `M1`, `M2`, `M3` 등 키
  - `name`, `program`, `settings`, `actions`
- `settings_main`
  - `resolution`, `custom`

액션 구조(대표):
- `A1` ~ `A6`
- 필드: `name`, `type(image/delay)`, `number`, `image`, `priority`, `enabled`, `coordinates`

관찰:
- `M1/M2/M3` 모두 OCR 모드 설정(`matcher_mode: ocr`) 사용
- 이미지 경로는 상대경로로 저장되고, 로딩 시 `resolve_project_path()`로 보정됨

## 6. 비전/좌표 계산 로직 핵심

### 6.1 템플릿 기반 (`core_functions/image_matcher.py`)
- `cv2.matchTemplate`로 A1 탐색
- `template_sizes`(원본 좌표 기반)와 실제 템플릿 크기 비율로 스케일 산출
- 액션 좌표는 `coordinates` 기반으로 환산해 중심점 클릭

### 6.2 OCR 기반 (`image_matcher_easyocr.py`)
- EasyOCR 싱글톤 리더 재사용
- 앱 식별 후 ROI 분기
  - 배민: 우하단(제4사분면)
  - 쿠팡: 우상단(제1사분면)
- `accept/reject` 텍스트 기반 탐색
- 시간 버튼은 텍스트 기준 오프셋 추정

### 6.3 파사드 (`core_functions/vision_engine.py`)
- 의도: OCR/Template 전략 전환
- 실제: 템플릿 ID가 `M\d+_A\d+` 패턴이면 강제로 템플릿 매처 경로 사용

## 7. UI 계층 정리

### 7.1 핵심 UI
- `MainWindow.ui`: 매크로 리스트/추가/삭제/복제/실행/중지/로그 패널
- `ActionWindow.ui`: 액션 테이블 편집, 순서 조정, 딜레이 추가, 프로그램 매핑
- `ActionWizardWindow.ui` + `Step2/Step3Window.ui`: 템플릿/영역 등록

### 7.2 구현 연결
- `dialog/*`가 실제 이벤트 처리 담당
- `ui/*_ui.py`는 생성 산출물(핵심 로직 없음)

## 8. 문서와 구현 일치도

### 일치하는 부분
- 경로 추상화(`path_manager`) 적용
- GUI 로그 패널 + F12 긴급중단 구현
- `VisionEngine` 추상화 도입
- Windows 빌드 준비물(`DeepOrder.spec`, 배치/PS 스크립트) 존재

### 차이가 있는 부분
- 문서에서 OCR 중심을 강조하나, 런타임 ID(`M*_A*`)는 템플릿 경로로 우회됨
- 일부 문서 수치(정확도/속도)는 실험 스크립트 성격이며 런타임에서 재현 보장은 별도 검증 필요

## 9. 핵심 리스크/결함 포인트

중요도 순:

1. `ActionSettingDialog.save_settings()` 미구현 (`pass`)
2. `VisionEngine` 모드 전환 실효성 제한 (`M*_A*` 템플릿 강제)
3. 좌표 보정 상수 `*2`, `//2` 혼재
4. 라이브러리 계층 `sys.exit()` 사용
5. 레거시/사장 코드 잔존 (`image_dialog` 등)
6. 경로/임포트 레거시 (`macro_scheduler`, `test_controller`)
7. 로깅 전략 혼재 (`print` + 파일 + GUI logger)

## 10. 테스트/증빙 상태

### 저장소 내 증빙
- `docs/portfolio/evidence/20260223_220503/summary.json`
- GUI 타임아웃/F12 로그 캡처 존재
- 샘플 이미지 기반 OCR 검증 결과 존재

### 관찰
- 실배달앱 라이브 환경보다 샘플/오프스크린 검증 중심
- 운영 신뢰도 확보에는 실환경 재검증 필요

## 11. 강점

- UI-실행엔진-데이터 계층 분리
- 매크로 CRUD 및 위저드 흐름 end-to-end 연결
- 경로 추상화 및 패키징 준비도 양호
- 실행 타임아웃/재시도/중단 체계 존재
- 실험 코드가 풍부하여 확장 방향 명확

## 12. 개선 우선순위 제안

1. 좌표 변환 규칙 통일 (`*2`/`//2` 제거 또는 단일화)
2. `VisionEngine` 정책 재정의 (OCR/Template 실제 분기)
3. 설정 UI 완성 (`ActionSettingDialog.save_settings`)
4. 종료/예외 처리 개선 (`sys.exit` → 예외/상태 반환)
5. 레거시 코드 정리 (`old/` 분리 또는 제거)

## 13. 결론

현재 DeepOrder는 **실제 동작 가능한 매크로 자동화 도구**로서 핵심 경로(생성-저장-실행-중지)는 유효하다. 다만 상용 안정성 관점에서는 **좌표 변환 일관성**, **매처 전략 실효성**, **설정 저장 미완성**, **종료/예외 처리**가 주요 리스크다.

즉, 기반은 충분히 좋고 확장 방향도 명확하지만, 운영 품질을 위해서는 위 4개 축을 우선 정리해야 한다.
