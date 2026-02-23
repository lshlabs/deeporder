param(
    [string]$PythonVersion = "3.11",
    [switch]$RunApp,
    [switch]$BuildExe
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Resolve-Python {
    param([string]$Version)

    $candidates = @(
        "py -$Version",
        "py",
        "python"
    )

    foreach ($candidate in $candidates) {
        try {
            & cmd /c "$candidate --version" | Out-Null
            return $candidate
        } catch {
            continue
        }
    }

    throw "Python launcher/python executable을 찾지 못했습니다. Python $Version 설치 후 다시 실행하세요."
}

function Ensure-Venv {
    param([string]$PythonCmd)

    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        Write-Step "가상환경 생성 (.venv)"
        & cmd /c "$PythonCmd -m venv .venv"
    } else {
        Write-Step "기존 가상환경 재사용 (.venv)"
    }
}

function Invoke-VenvPython {
    param([string]$ArgsLine)
    & cmd /c ".venv\Scripts\python.exe $ArgsLine"
}

Write-Step "DeepOrder Windows 개발환경 셋업 시작"

if (-not (Test-Path "main.py")) {
    throw "저장소 루트에서 실행해주세요. (main.py 없음)"
}

$pythonCmd = Resolve-Python -Version $PythonVersion
Write-Host "Python command: $pythonCmd" -ForegroundColor DarkGray

Ensure-Venv -PythonCmd $pythonCmd

Write-Step "pip 업그레이드"
Invoke-VenvPython '-m pip install --upgrade pip'

Write-Step "의존성 설치 (requirements-dev.txt)"
Invoke-VenvPython '-m pip install -r requirements-dev.txt'

Write-Step "기본 import/경로 스모크 체크"
Invoke-VenvPython '-c "from utils.path_manager import ui_path, data_path; print(\"UI:\", ui_path(\"MainWindow.ui\").exists()); print(\"DATA:\", data_path().exists())"'

if ($RunApp) {
    Write-Step "앱 실행 (main.py)"
    Invoke-VenvPython 'main.py'
}

if ($BuildExe) {
    Write-Step "PyInstaller 빌드 (DeepOrder.spec)"
    & cmd /c ".venv\Scripts\pyinstaller.exe --noconfirm --clean DeepOrder.spec"
}

Write-Step "완료"
Write-Host "앱 실행: .venv\Scripts\python.exe main.py" -ForegroundColor Green
Write-Host "빌드 실행: scripts\build_windows.bat  또는  .venv\Scripts\pyinstaller.exe --noconfirm --clean DeepOrder.spec" -ForegroundColor Green
