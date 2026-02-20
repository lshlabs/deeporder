# DeepOrder 핵심기능 코드 + 포트폴리오 문장

아래 6개는 현재 코드베이스에서 확인되는 핵심 기능입니다.  
프로젝트가 미완성 단계인 점을 반영해, **실제로 동작 중인 기능 중심**으로 정리했습니다.

## 1) 매크로 스레드 실행/중지 관리

**코드 위치**  
- `core_functions/macro_runner.py:27`  
- `core_functions/macro_runner.py:54`

**문제/해결**  
- 문제: GUI가 멈추지 않으면서 매크로를 실행/중지해야 함.  
- 해결: 매크로별 스레드와 `Event` 기반 stop flag를 두어 비동기 실행 제어.

**핵심 코드**
```python
def start_macro(self, macro_key):
    if macro_key in self.running_macros and self.running_macros[macro_key].is_alive():
        return False
    self.stop_flags[macro_key] = threading.Event()
    thread = threading.Thread(target=self._run_macro, args=(macro_key, macro_data, self.stop_flags[macro_key]))
    thread.daemon = True
    thread.start()
    self.running_macros[macro_key] = thread
    self.status_changed.emit(macro_key, "running")
    return True

def stop_macro(self, macro_key):
    if macro_key in self.stop_flags:
        self.stop_flags[macro_key].set()
        self.status_changed.emit(macro_key, "stopped")
        return True
    return False
```

**이력서/포트폴리오 문장**  
- 매크로 실행 엔진을 스레드 기반으로 구성해 UI 블로킹 없이 실행/중지를 처리했습니다.  
- 매크로별 stop flag를 적용해 동시 실행 제어와 안전한 종료 흐름을 구현했습니다.

## 2) 템플릿 매칭 + 스케일 보정 좌표 계산

**코드 위치**  
- `core_functions/image_matcher.py:108`  
- `core_functions/image_matcher.py:142`  
- `core_functions/image_matcher.py:160`

**문제/해결**  
- 문제: 템플릿과 실제 화면 크기가 다르면 액션 좌표가 어긋날 수 있음.  
- 해결: 템플릿 매칭 후 원본 좌표 대비 `scale_x/scale_y`를 계산해 액션 좌표를 보정.

**핵심 코드**
```python
result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)
if max_val < self.threshold:
    return False, None, max_val, screenshot, None

scale_x = template_w / orig_width if orig_width > 0 else 1.0
scale_y = template_h / orig_height if orig_height > 0 else 1.0
scaled_x = int(orig_x * scale_x)
scaled_y = int(orig_y * scale_y)
abs_x = template_location[0] // 2 + scaled_x
abs_y = template_location[1] // 2 + scaled_y
```

**이력서/포트폴리오 문장**  
- OpenCV 템플릿 매칭 결과에 스케일 보정을 적용해 액션 클릭 좌표 오차를 줄였습니다.  
- 템플릿-액션 좌표 변환 로직을 모듈화해 클릭 타겟 계산을 재사용 가능하게 구성했습니다.

## 3) 위저드 기반 액션 세트 자동 생성(A1~A6)

**코드 위치**  
- `utils/data_manager.py:60`  
- `utils/data_manager.py:133`

**문제/해결**  
- 문제: 매크로를 수동으로 등록하면 액션 구성 누락/번호 충돌이 발생하기 쉬움.  
- 해결: 위저드 이미지와 드래그 좌표를 바탕으로 표준 액션 세트를 자동 생성하고 JSON에 저장.

**핵심 코드**
```python
mapping = {
    'step1': ('원본 이미지', 'A1.png'),
    'minus': ('- 버튼 이미지', 'A2.png'),
    'plus': ('+ 버튼 이미지', 'A3.png'),
    'time': ('예상시간 이미지', 'A4.png'),
    'reject': ('거부버튼 이미지', 'A5.png'),
    'accept': ('접수버튼 이미지', 'A6.png')
}
actions = self._wizard_actions_common(macro_key, mapping, starting_action_number=1)
self._data['macro_list'][macro_key]['actions'] = actions
self.save_data()
```

**이력서/포트폴리오 문장**  
- 매크로 생성 시 A1~A6 액션 구조를 자동 생성해 초기 설정 시간을 단축했습니다.  
- 좌표/이미지/순번 정보를 JSON으로 표준화해 매크로 데이터 일관성을 유지했습니다.

## 4) 매크로 복제 + 이미지 자산 경로 재매핑

**코드 위치**  
- `utils/data_manager.py:260`  
- `utils/data_manager.py:290`

**문제/해결**  
- 문제: 매크로를 복제할 때 이미지 폴더와 경로를 함께 복제하지 않으면 깨진 참조가 발생함.  
- 해결: 매크로 키 재할당 후 폴더 복사와 액션별 이미지 경로를 새 폴더 기준으로 재작성.

**핵심 코드**
```python
new_macro_key = f"M{next_num}"
new_macro = {'name': new_name, 'program': original_macro.get('program'), 'actions': {}}
for action_key, action_data in original_macro['actions'].items():
    new_macro['actions'][action_key] = action_data.copy()

shutil.copytree(original_folder, new_folder)
for action in new_macro['actions'].values():
    if action['type'] == 'image':
        old_path = Path(action['image'])
        action['image'] = str(new_folder / old_path.name)
```

**이력서/포트폴리오 문장**  
- 매크로 복제 기능에 자산 폴더 복사와 경로 재매핑을 포함해 즉시 재사용 가능한 복제 흐름을 구현했습니다.  
- 새 매크로 키 자동 할당으로 기존 데이터와 충돌 없이 복제본을 생성하도록 처리했습니다.

## 5) EasyOCR 싱글톤 모델 재사용 + 앱별 ROI 탐지

**코드 위치**  
- `image_matcher_easyocr.py:40`  
- `image_matcher_easyocr.py:56`  
- `image_matcher_easyocr.py:73`  
- `image_matcher_easyocr.py:143`

**문제/해결**  
- 문제: OCR 모델을 매번 초기화하면 지연이 커지고, 전체 화면 OCR은 비용이 큼.  
- 해결: EasyOCR 리더를 싱글톤으로 재사용하고, 배민/쿠팡 앱별 ROI + 키워드 규칙으로 탐지 범위를 축소.

**핵심 코드**
```python
if ImageMatcherEasyOCR._reader is None:
    ImageMatcherEasyOCR._reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
self.reader = ImageMatcherEasyOCR._reader

self.delivery_app_config = {
    'baemin_quadrant4_roi': {'x_start_ratio': 0.5, 'y_start_ratio': 0.5, 'x_end_ratio': 1.0, 'y_end_ratio': 1.0},
    'coupang_quadrant1_roi': {'x_start_ratio': 0.5, 'y_start_ratio': 0.0, 'x_end_ratio': 1.0, 'y_end_ratio': 0.5},
}
```

**이력서/포트폴리오 문장**  
- EasyOCR 리더 싱글톤 구조를 적용해 반복 실행 시 초기화 비용을 줄였습니다.  
- 앱별 ROI/키워드 기반 탐지 규칙을 도입해 OCR 검색 범위를 최적화했습니다.

## 6) OCR 패턴 매칭 기반 실시간 모니터링 파이프라인

**코드 위치**  
- `ocr_test.py:29`  
- `ocr_test.py:92`  
- `ocr_test.py:167`

**문제/해결**  
- 문제: 단순 키워드 감지만으로는 버튼/시간 요소를 구조적으로 분리하기 어려움.  
- 해결: `exact/regex` 패턴 매칭, 박스 시각화, 결과 이미지 저장까지 한 파이프라인으로 구성.

**핵심 코드**
```python
BAEMIN_PATTERNS = {
    "accept_button": {"type": "exact", "values": ["접수"]},
    "reject_button": {"type": "exact", "values": ["거부"]},
    "time": {"type": "regex", "values": [r"\d{1,2}~\d{1,2}분", r"\d{1,2}\s*분"]},
}

ocr_entries = run_ocr(reader, frame)
element_matches = collect_matches(ocr_entries, BAEMIN_PATTERNS)
annotated = annotate_image(frame, matches_baemin_keyword, element_matches, "baemin")
filepath = save_result(annotated, output_dir, "baemin")
```

**이력서/포트폴리오 문장**  
- OCR 결과를 exact/regex 규칙으로 후처리해 버튼/시간 요소를 구조화된 데이터로 추출했습니다.  
- 감지 결과를 주석 이미지로 저장해 디버깅과 성능 비교가 가능한 검증 루프를 만들었습니다.

