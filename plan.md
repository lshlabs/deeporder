# DeepOrder 전면 재설계 계획서 (기획문서 기준, Windows 우선)

## 요약
본 계획은 기존 DeepOrder를 “템플릿 매칭 기반 클릭 자동화”에서 “조건 트리거 + 프리셋 규칙 엔진 기반 자동화 플랫폼”으로 전면 개편한다.  
핵심 목표는 다음 3가지다.

1. 세팅 단계 재구성: 감시모드 캡처 → 포인트 좌표 매핑 → 트리거/버튼/중요텍스트/프리셋 구성.
2. 동작 단계 재구성: 속도 우선(컬러 트리거 1차, OCR 2차), 조건 기반 매크로/프리셋 라우팅.
3. 기존 구조 보관: 삭제 대신 `old/`로 이동 후 신규 구조를 독립 구축.

---

## 1) 기획 요구사항 반영 범위

1. 세팅 1단계
- 매크로 생성 시 감시모드 진입.
- 주문 팝업 포착 시 전체화면 캡처.
- 사용자가 클릭 대상/중요텍스트 지점을 점으로 추가.
- 각 점에 대해 `[스크린샷 좌표 : 실제 화면 좌표]` 1:1 저장.

2. 세팅 2단계
- 트리거를 매크로 상세/수정에서 설정.
- 지원 트리거:
  - 픽셀 컬러 트리거 (`(x,y)`에서 특정 HEX 등장)
  - OCR 텍스트 트리거 (`ROI`에서 특정 문자열 포착)

3. 세팅 3단계
- 점을 `button`과 `important_text`로 구분.
- `button`은 클릭 순서/횟수 설정.
- `important_text`는 분석용(클릭 없음), OCR 판독 대상.

4. 세팅 4단계
- 프리셋(동작 시퀀스) 다중 생성.
- 프리셋마다 조건 부여:
  - 예: 중요텍스트가 `15분`이면 프리셋1(`+` 1회)
  - 예: 중요텍스트가 `10분`이면 프리셋2(`+` 2회)

5. 동작 단계
- 핫키로 실행/중단.
- 성공/실패 횟수 및 로그 누적.
- 실패 시 스크린샷 자동 저장.
- 조건에 따라 매크로 선택 + 조건에 따라 프리셋 선택.

---

## 2) 현재 구현 대비 핵심 차이와 전환 원칙

1. 기존 A1~A6 고정 액션 구조 폐지.
2. 템플릿 중심 탐색을 트리거 중심 탐색으로 전환.
3. `matcher_mode` 형식적 옵션이 아닌 실제 엔진 분기 구조로 교체.
4. 좌표 보정 `*2`, `//2` 관행 제거, 단일 좌표계(`screen_xy`) 확정.
5. 라이브러리 내부 `sys.exit()` 금지, 예외/상태 반환으로 상위 제어.
6. 기존 코드는 삭제하지 않고 `old/` 이동.

---

## 3) 목표 아키텍처 (Decision Complete)

런타임 파이프라인:

`ScreenCapture` → `TriggerEngine` → `MacroRouter` → `PresetResolver` → `ActionExecutor` → `EvidenceLogger`

### 구성요소 책임
1. `ScreenCapture`
- 프레임 캡처, ROI 캐싱, 주기 제어(FPS).

2. `TriggerEngine`
- 컬러/OCR 트리거 평가.
- 속도 정책: 컬러 우선, OCR 후순위.

3. `MacroRouter`
- 다중 매크로 중 실행 후보 결정(우선순위 기반).

4. `PresetResolver`
- 중요텍스트 판독값으로 프리셋 조건 매칭.

5. `ActionExecutor`
- 클릭/다중클릭/딜레이 실행.

6. `EvidenceLogger`
- 성공/실패 카운터, 실행 로그, 실패 스크린샷 저장.

---

## 4) 신규 도메인 모델/API

```python
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List, Dict

PointType = Literal["button", "important_text"]
TriggerType = Literal["pixel_color", "ocr_text"]
ActionType = Literal["click", "click_n", "delay", "read_text"]

@dataclass
class ScreenPoint:
    point_id: str
    name: str
    point_type: PointType
    screenshot_xy: Tuple[int, int]
    screen_xy: Tuple[int, int]
    roi: Optional[Tuple[int, int, int, int]] = None

@dataclass
class Trigger:
    trigger_id: str
    trigger_type: TriggerType
    enabled: bool
    config: Dict

@dataclass
class ActionStep:
    step_id: str
    action_type: ActionType
    target_point_id: Optional[str] = None
    click_count: int = 1
    delay_ms: int = 0

@dataclass
class PresetRule:
    rule_id: str
    important_point_id: str
    operator: Literal["equals", "contains", "regex"]
    value: str

@dataclass
class Preset:
    preset_id: str
    name: str
    priority: int
    conditions: List[PresetRule]
    steps: List[ActionStep]

@dataclass
class Macro:
    macro_id: str
    name: str
    enabled: bool
    triggers: List[Trigger]
    points: List[ScreenPoint]
    presets: List[Preset]
```

---

## 5) 저장 스키마(v2)

```json
{
  "version": 2,
  "platform": "windows",
  "hotkeys": {
    "start_stop_toggle": "F8",
    "panic_stop": "F12"
  },
  "runtime": {
    "capture_fps": 10,
    "trigger_logic": "ANY",
    "color_first_timeout_ms": 20,
    "ocr_fallback_timeout_ms": 120
  },
  "macros": [
    {
      "macro_id": "macro_001",
      "name": "배민 기본",
      "enabled": true,
      "priority": 100,
      "triggers": [],
      "points": [],
      "presets": []
    }
  ]
}
```

---

## 6) 코드 구조 재편 계획

### 기존 구조 처리
- 기존 런타임/실험/UI 관련 코드는 `old/`로 이동.
- 예시:
  - `old/dialog/`
  - `old/core_functions/`
  - `old/utils/`
  - `old/tests/`
  - `old/experiments/`
  - `old/ui/legacy/`
  - `old/utils/data_legacy.json`

### 신규 구조
- `app/bootstrap.py`
- `app/domain/models.py`
- `app/engine/trigger_engine.py`
- `app/engine/macro_router.py`
- `app/engine/preset_resolver.py`
- `app/engine/action_executor.py`
- `app/services/setup_session.py`
- `app/services/runtime_service.py`
- `app/storage/repository.py`
- `app/adapters/screen_capture.py`
- `app/adapters/ocr_adapter.py`
- `app/adapters/color_probe.py`
- `app/hotkey/windows_hotkey.py`
- `app/logging/evidence_logger.py`
- `app/ui/main_window.py`
- `app/ui/setup_wizard.py`
- `app/ui/macro_editor.py`

---

## 7) 핵심 엔진 구현 스니펫

### TriggerEngine
```python
class TriggerEngine:
    def __init__(self, color_eval, ocr_eval):
        self.color_eval = color_eval
        self.ocr_eval = ocr_eval

    def evaluate_macro(self, macro, frame):
        results = []
        for t in macro.triggers:
            if not t.enabled:
                continue
            if t.trigger_type == "pixel_color":
                results.append(self.color_eval.eval(t.config, frame))
            elif t.trigger_type == "ocr_text":
                results.append(self.ocr_eval.eval_contains(t.config, frame))
        return any(results)  # v2 기본: ANY
```

### PresetResolver
```python
import re

class PresetResolver:
    def resolve(self, macro, text_values):
        candidates = sorted(macro.presets, key=lambda p: p.priority, reverse=True)
        for preset in candidates:
            ok = True
            for c in preset.conditions:
                actual = text_values.get(c.important_point_id, "")
                if c.operator == "equals" and actual != c.value:
                    ok = False
                elif c.operator == "contains" and c.value not in actual:
                    ok = False
                elif c.operator == "regex" and re.search(c.value, actual) is None:
                    ok = False
            if ok:
                return preset
        return None
```

### ActionExecutor
```python
class ActionExecutor:
    def __init__(self, mouse, sleeper):
        self.mouse = mouse
        self.sleeper = sleeper

    def run(self, preset, point_map):
        for step in preset.steps:
            if step.action_type == "click":
                x, y = point_map[step.target_point_id]
                self.mouse.click(x, y)
            elif step.action_type == "click_n":
                x, y = point_map[step.target_point_id]
                for _ in range(step.click_count):
                    self.mouse.click(x, y)
            elif step.action_type == "delay":
                self.sleeper.ms(step.delay_ms)
        return True
```

---

## 8) UI/UX 상세 동작 정의

1. 메인 화면
- 매크로 목록, 상태(활성/비활성), 성공/실패 카운트.
- 전역 핫키 상태 표시.
- 실시간 로그 패널.

2. 매크로 생성 위저드
- Step A: 감시모드 시작/중지.
- Step B: 팝업 포착 후 전체화면 캡처.
- Step C: 포인트 추가/수정/삭제 및 타입 지정(button/important_text).
- Step D: 트리거 설정.
- Step E: 프리셋 조건+스텝 구성.

3. 매크로 상세/수정
- 트리거 편집.
- 버튼 클릭 순서 시각화.
- 중요텍스트 OCR ROI 미리보기.
- 프리셋별 우선순위 설정.

---

## 9) 테스트/검증 계획

1. 단위 테스트
- 트리거 평가 정확성(컬러/OCR).
- 프리셋 조건 매칭 정확성.
- 액션 시퀀스 실행 순서/횟수 검증.

2. 통합 테스트
- 캡처→트리거→매크로 선택→프리셋 선택→실행 end-to-end.
- 실패 시 증적 저장(이미지+로그) 확인.

3. 성능 테스트
- 트리거 평가 지연(ms) 측정.
- 매크로 라우팅 지연 측정.
- OCR fallback 비율 측정.

4. 회귀 테스트
- `old/` 이동 후 import 충돌 없음.
- 신규 스키마 로드/저장 안정성 확인.

---

## 10) 단계별 실행 로드맵

1. Phase 1: 구조 재편
- `old/` 이동 + 신규 골격 생성 + 부팅 확인.

2. Phase 2: 세팅 단계 완성
- 감시모드/캡처/포인트 매핑/트리거 편집 구현.

3. Phase 3: 동작 엔진 완성
- TriggerEngine, Router, PresetResolver, Executor 구현.

4. Phase 4: 핫키/로깅/증적
- Windows 글로벌 핫키 + 실패 스크린샷 + 통계 로그.

5. Phase 5: 안정화
- 통합 테스트, 성능 튜닝, 문서 정리.

---

## 11) 주요 가정/기본값

1. 플랫폼은 Windows 우선.
2. 트리거는 혼합형(컬러+OCR), 기본 결합은 `ANY`.
3. 기존 구조는 유지하되 신규 런타임에서 미사용(`old/` 보관).
4. 좌표계는 `screen_xy` 단일 기준.
5. 핫키 기본값은 `F8`(토글), `F12`(패닉중단).
