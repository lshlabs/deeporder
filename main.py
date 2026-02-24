import os


def preload_torch_for_windows() -> None:
    """Windows에서 torch를 먼저 로드해서 DLL 초기화 오류를 줄인다."""
    if os.name != "nt":
        return

    try:
        import torch  # noqa: F401
    except Exception:
        # 여기서 실패해도 나중에 실제 에러 메시지가 다시 나오게 둔다.
        pass


if __name__ == "__main__":
    preload_torch_for_windows()
    from dialog.main_dialog import main

    main()
