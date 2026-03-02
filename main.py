import os


def preload_torch_for_windows() -> None:
    """Windows에서 torch를 먼저 불러서 초기 DLL 오류를 줄입니다."""
    if os.name != "nt":
        return

    try:
        import torch  # noqa: F401
    except Exception:
        # 선로딩이 실패해도 실제 실행 시 원래 오류를 다시 보여줍니다.
        pass


if __name__ == "__main__":
    preload_torch_for_windows()
    from dialog.main_dialog import main

    main()
