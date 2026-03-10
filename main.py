import os
import logging


logger = logging.getLogger(__name__)


def preload_torch_for_windows() -> None:
    """윈도우에서 DLL 오류를 초기에 노출하기 위해 torch를 선로딩한다."""
    if os.name != "nt":
        return

    try:
        import torch  # noqa: F401
    except (ImportError, OSError) as exc:
        logger.warning("torch preload skipped: %s", exc)


if __name__ == "__main__":
    preload_torch_for_windows()
    from dialog.main_dialog import main

    main()
