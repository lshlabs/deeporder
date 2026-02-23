from __future__ import annotations

import logging
from collections import deque

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QTextCursor


class _LogSignal(QObject):
    update_log = pyqtSignal(str)


class GUILogHandler(logging.Handler):
    def __init__(self, text_widget, max_lines: int = 500):
        super().__init__()
        self.text_widget = text_widget
        self.max_lines = max_lines
        self._lines = deque(maxlen=max_lines)
        self.signals = _LogSignal()
        self.signals.update_log.connect(self._append_line)

    def emit(self, record):
        try:
            self.signals.update_log.emit(self.format(record))
        except Exception:
            self.handleError(record)

    def _append_line(self, line: str):
        self._lines.append(line)
        self.text_widget.setPlainText("\n".join(self._lines))
        cursor = self.text_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_widget.setTextCursor(cursor)


def bind_text_widget(widget, logger_name: str = "deeporder.ui", max_lines: int = 500) -> GUILogHandler:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = GUILogHandler(widget, max_lines=max_lines)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return handler
