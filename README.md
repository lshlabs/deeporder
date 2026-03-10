# DeepOrder

DeepOrder는 화면 좌표 기반으로 매크로를 생성하고, 트리거 조건이 맞을 때 자동으로 실행하는 Windows 데스크톱 자동화 도구입니다.  
PyQt6 기반 GUI 앱이며, 매크로 생성, 상세 편집, 트리거 설정, 감시 실행까지 하나의 흐름으로 구성되어 있습니다.

## 주요 기능

- 전체 화면 캡처를 기반으로 매크로 생성
- 버튼 영역 / 텍스트 영역 저장
- 매크로 트리거 / 프리셋 트리거 설정
- 감시모드에서 트리거 감지 후 자동 실행
- 프리셋, 대기시간, 메모, 반복횟수, 반복 딜레이 설정

## 기본 사용 흐름

1. 앱 실행
2. `설정`에서 해상도 및 핫키 설정
3. `생성`으로 매크로 생성 진입
4. 화면 캡처 후 영역 선택
5. 상세 편집에서 시퀀스와 트리거 설정
6. `RUN`으로 감시모드 실행

## 디렉터리 구조

- [`main.py`](C:/Users/Admin/Documents/deeporder/main.py): 앱 진입점
- [`dialog`](C:/Users/Admin/Documents/deeporder/dialog): PyQt 다이얼로그와 화면 제어 코드
- [`core_functions`](C:/Users/Admin/Documents/deeporder/core_functions): 실행 엔진, 트리거 판정, 마우스 제어
- [`utils`](C:/Users/Admin/Documents/deeporder/utils): 데이터 저장, 경로 처리, 로깅, 공용 유틸
- [`ui`](C:/Users/Admin/Documents/deeporder/ui): Qt Designer `.ui` 파일과 UI 리소스
- [`img`](C:/Users/Admin/Documents/deeporder/img): 사용자 매크로 이미지, 크롭 미리보기
- [`dist/DeepOrder`](C:/Users/Admin/Documents/deeporder/dist/DeepOrder): 현재 검증 완료된 최신 Windows 배포본

## 개발 환경

- Windows 10 이상
- Python 3.11
- 가상환경 사용 권장

의존성 파일:

- [`requirements.txt`](C:/Users/Admin/Documents/deeporder/requirements.txt)
- [`requirements-dev.txt`](C:/Users/Admin/Documents/deeporder/requirements-dev.txt)

## 로컬 실행 방법

가상환경 생성 및 활성화:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

앱 실행:

```powershell
.venv\Scripts\python.exe main.py
```

## 데이터 및 실행 경로 정책

DeepOrder는 정적 리소스와 사용자 데이터를 분리해서 다룹니다.

- 정적 리소스(`.ui`, 번들 리소스)는 번들 리소스 경로에서 읽습니다.
- 사용자 데이터(`data.json`, `img`)는 실행 파일 기준 외부 경로에서 읽고 씁니다.

중요 파일:

- [`utils/data.json`](C:/Users/Admin/Documents/deeporder/utils/data.json): 매크로 및 앱 설정 데이터
- [`img`](C:/Users/Admin/Documents/deeporder/img): 매크로 이미지와 캡처 결과

배포본에서는 `_internal` 내부 데이터가 아니라, 배포 폴더 바깥의 실제 `utils/data.json`, `img`를 기준으로 동작하도록 정리되어 있습니다.

## 빌드 및 배포

현재 검증 완료된 배포 실행 파일:

- [`dist/DeepOrder/DeepOrder.exe`](C:/Users/Admin/Documents/deeporder/dist/DeepOrder/DeepOrder.exe)

배포 방식:

- `onedir`
- `console=False`

spec 파일:

- [`DeepOrder.spec`](C:/Users/Admin/Documents/deeporder/DeepOrder.spec)

빌드 명령:

```powershell
cmd /c .venv\Scripts\pyinstaller.exe --noconfirm --clean DeepOrder.spec
```

## 검증 완료 상태

아래 흐름은 회귀 테스트를 완료했습니다.

- `설정 -> 생성 -> 상세 편집 -> 트리거 설정 -> RUN`