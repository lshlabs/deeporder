from __future__ import annotations
import copy

from PyQt6 import QtCore, QtGui, QtWidgets, uic

from dialog.trigger_dialog import TriggerDialog
from utils.data_manager import DataManager
from utils.path_manager import ui_path


class PresetTabLabel(QtWidgets.QPushButton):
    double_clicked = QtCore.pyqtSignal()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


class PresetRibbon(QtWidgets.QFrame):
    tab_selected = QtCore.pyqtSignal(str)
    tab_close_requested = QtCore.pyqtSignal(str)
    tab_rename_requested = QtCore.pyqtSignal(str)
    add_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("preset_ribbon")
        self._chip_height = 28
        self.setFixedHeight(self._chip_height)
        self.setStyleSheet(
            "QFrame#preset_ribbon { background: transparent; border: none; }"
            "QFrame#preset_chip {"
            "  background: #e4e4e4;"
            "  border: 1px solid #8e8e8e;"
            "  border-bottom: none;"
            "  border-top-left-radius: 7px;"
            "  border-top-right-radius: 7px;"
            "  border-bottom-left-radius: 0px;"
            "  border-bottom-right-radius: 0px;"
            "}"
            "QFrame#preset_chip:hover { background: #e9eef5; border-color: #7d8ea3; border-bottom: none; }"
            "QFrame#preset_chip[selected='true'] {"
            "  background: #ffffff;"
            "  border-color: #2f2f2f;"
            "  border-bottom: none;"
            "}"
            "QPushButton#preset_label {"
            "  border: none;"
            "  background: transparent;"
            "  color: #111827;"
            "  padding: 0 10px;"
            "  min-height: 22px;"
            "}"
            "QPushButton#preset_label:hover { color: #0f172a; }"
            "QPushButton#preset_close {"
            "  min-width: 13px; max-width: 13px;"
            "  min-height: 13px; max-height: 13px;"
            "  padding: 0px;"
            "  border-radius: 6px;"
            "  border: none;"
            "  background: transparent;"
            "  color: #6b7280;"
            "}"
            "QPushButton#preset_close:hover { background: #d7dce3; color: #111827; }"
            "QPushButton#preset_add {"
            "  min-width: 28px; max-width: 28px;"
            "  min-height: 28px; max-height: 28px;"
            "  padding: 0px;"
            "  margin: 0px;"
            "  border-top-left-radius: 7px;"
            "  border-top-right-radius: 7px;"
            "  border-bottom-left-radius: 0px;"
            "  border-bottom-right-radius: 0px;"
            "  border: 1px solid #8e8e8e;"
            "  border-bottom: none;"
            "  background: #e4e4e4;"
            "  color: #111827;"
            "}"
            "QPushButton#preset_add:hover { background: #e9eef5; border-color: #7d8ea3; border-bottom: none; }"
        )
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom)
        self._layout = layout
        self._widgets = []

    def _clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._widgets.clear()

    def set_tabs(self, tabs: list[dict], current_preset_id: str | None, default_preset_id: str | None):
        self._clear()
        for tab in tabs:
            preset_id = tab["id"]
            chip = QtWidgets.QFrame(self)
            chip.setObjectName("preset_chip")
            chip.setProperty("selected", preset_id == current_preset_id)
            chip.setFixedHeight(self._chip_height)

            chip_layout = QtWidgets.QHBoxLayout(chip)
            chip_layout.setContentsMargins(0, 0, 5, 0)
            chip_layout.setSpacing(3)
            chip_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

            label = PresetTabLabel(tab["name"], chip)
            label.setObjectName("preset_label")
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
            label.setMinimumHeight(22)
            label.setMaximumHeight(22)
            label.clicked.connect(lambda _=False, pid=preset_id: self.tab_selected.emit(pid))
            label.double_clicked.connect(lambda pid=preset_id: self.tab_rename_requested.emit(pid))
            chip_layout.addWidget(label)

            if preset_id != default_preset_id:
                close_btn = QtWidgets.QPushButton("x", chip)
                close_btn.setObjectName("preset_close")
                close_btn.clicked.connect(lambda _=False, pid=preset_id: self.tab_close_requested.emit(pid))
                chip_layout.addWidget(close_btn)

            chip.style().unpolish(chip)
            chip.style().polish(chip)
            self._layout.addWidget(chip, 0)
            self._widgets.append(chip)

        add_btn = QtWidgets.QPushButton("+", self)
        add_btn.setObjectName("preset_add")
        add_btn.setFixedHeight(self._chip_height)
        add_btn.clicked.connect(self.add_requested.emit)
        self._layout.addWidget(add_btn, 0)
        self._layout.addStretch(1)


class ItemEditDialog(QtWidgets.QDialog):
    def __init__(self, item: dict, click_count: int | None = None, parent=None):
        super().__init__(parent)
        self.item = item
        self._build_ui(click_count)

    def _build_ui(self, click_count: int | None):
        uic.loadUi(str(ui_path("ItemEditDialog.ui")), self)
        self.setWindowTitle("항목 수정")
        self.setModal(True)
        target_height = 500 if self.item.get("item_type") != "text" else 450
        self.resize(max(self.width(), 460), target_height)

        self.preview_label = self.findChild(QtWidgets.QLabel, "preview_label")
        self.line_name = self.findChild(QtWidgets.QLineEdit, "line_name")
        self.spin_x = self.findChild(QtWidgets.QSpinBox, "spin_x")
        self.spin_y = self.findChild(QtWidgets.QSpinBox, "spin_y")
        self.spin_w = self.findChild(QtWidgets.QSpinBox, "spin_w")
        self.spin_h = self.findChild(QtWidgets.QSpinBox, "spin_h")
        self.spin_clicks = self.findChild(QtWidgets.QSpinBox, "spin_clicks")
        self.clicks_label = self.findChild(QtWidgets.QLabel, "clicks_label")
        self.button_save = self.findChild(QtWidgets.QPushButton, "button_save")
        self.button_cancel = self.findChild(QtWidgets.QPushButton, "button_cancel")

        preview_path = self.item.get("preview_image")
        if preview_path:
            pixmap = QtGui.QPixmap(preview_path)
            if not pixmap.isNull():
                preview_size = self.preview_label.size()
                scaled = pixmap.scaled(
                    max(160, preview_size.width() - 12),
                    max(96, preview_size.height() - 12),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(scaled)

        rect = self.item.get("screen_rect") or self.item.get("capture_rect")
        for spin in (self.spin_x, self.spin_y, self.spin_w, self.spin_h):
            spin.setRange(0, 99999)
            spin.setMinimumHeight(34)

        if rect:
            self.spin_x.setValue(int(rect.get("x", 0) or 0))
            self.spin_y.setValue(int(rect.get("y", 0) or 0))
            self.spin_w.setValue(int(rect.get("width", 0) or 0))
            self.spin_h.setValue(int(rect.get("height", 0) or 0))

        self.line_name.setText(self.item.get("name", ""))
        if self.item.get("item_type") != "text":
            self.spin_clicks.setRange(1, 99)
            self.spin_clicks.setValue(max(1, int(click_count or 1)))
        else:
            self.clicks_label.hide()
            self.spin_clicks.hide()
            self.resize(max(self.width(), 460), 450)

        self.button_save.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

    def get_values(self):
        name = self.line_name.text().strip() or self.item.get("name", "")
        clicks = self.spin_clicks.value() if self.spin_clicks is not None else None
        rect = {
            "x": self.spin_x.value(),
            "y": self.spin_y.value(),
            "width": self.spin_w.value(),
            "height": self.spin_h.value(),
        }
        return name, clicks, rect


class MacroSettingsDialog(QtWidgets.QDialog):
    def __init__(self, macro_settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("매크로 설정")
        self.setModal(True)
        self.resize(320, 180)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        repeat_row = QtWidgets.QHBoxLayout()
        repeat_label = QtWidgets.QLabel("실행 횟수")
        repeat_label.setFixedWidth(80)
        self.spin_repeat = QtWidgets.QSpinBox(self)
        self.spin_repeat.setRange(0, 9999)
        self.spin_repeat.setValue(int(macro_settings.get("repeat_count", 1) or 0))
        repeat_hint = QtWidgets.QLabel("0 = 무한 반복")
        repeat_hint.setStyleSheet("color:#4b5563;")
        repeat_row.addWidget(repeat_label)
        repeat_row.addWidget(self.spin_repeat)
        repeat_row.addWidget(repeat_hint, 1)
        root.addLayout(repeat_row)

        delay_row = QtWidgets.QHBoxLayout()
        delay_label = QtWidgets.QLabel("반복 딜레이")
        delay_label.setFixedWidth(80)
        self.spin_delay = QtWidgets.QDoubleSpinBox(self)
        self.spin_delay.setRange(0.0, 9999.0)
        self.spin_delay.setDecimals(1)
        self.spin_delay.setSingleStep(0.1)
        self.spin_delay.setValue(float(macro_settings.get("repeat_delay_sec", 0.5) or 0.0))
        delay_suffix = QtWidgets.QLabel("초")
        delay_row.addWidget(delay_label)
        delay_row.addWidget(self.spin_delay)
        delay_row.addWidget(delay_suffix)
        delay_row.addStretch(1)
        root.addLayout(delay_row)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.button_save = QtWidgets.QPushButton("저장", self)
        self.button_cancel = QtWidgets.QPushButton("취소", self)
        footer.addWidget(self.button_save)
        footer.addWidget(self.button_cancel)
        root.addLayout(footer)

        self.button_save.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

    def values(self):
        return self.spin_repeat.value(), self.spin_delay.value()


class ActionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, macro_key=""):
        super().__init__(parent)
        self.macro_key = macro_key
        self.data_manager = DataManager.get_instance()
        self.macro_data = self.data_manager.get_macro(self.macro_key)
        self._macro_snapshot = copy.deepcopy(self.macro_data) if self.macro_data else None
        self._committed = False
        self._copied_step = None
        self.current_preset_id = self.macro_data.get("default_preset_id")
        self.content_container = None
        self.preset_ribbon = None
        self._shortcuts = []
        self._build_ui()
        self._connect_signals()
        self.reload_everything()

    def _build_ui(self):
        uic.loadUi(str(ui_path("ActionDialog.ui")), self)
        self.card = self.findChild(QtWidgets.QFrame, "card")
        self.tab_widget = self.findChild(QtWidgets.QTabWidget, "tab_widget")
        self.list_card = self.findChild(QtWidgets.QFrame, "list_card")
        self.table_widget = self.findChild(QtWidgets.QTableWidget, "table_widget")
        self.panel_card = self.findChild(QtWidgets.QFrame, "panel_card")
        self.action_card = self.findChild(QtWidgets.QFrame, "action_card")
        self.move_card = self.findChild(QtWidgets.QFrame, "move_card")
        self.button_add = self.findChild(QtWidgets.QPushButton, "button_add")
        self.button_delay = self.findChild(QtWidgets.QPushButton, "button_delay")
        self.button_note = self.findChild(QtWidgets.QPushButton, "button_note")
        self.button_priority = self.findChild(QtWidgets.QPushButton, "button_priority")
        self.button_program = self.findChild(QtWidgets.QPushButton, "button_program")
        self.button_up = self.findChild(QtWidgets.QPushButton, "button_up")
        self.button_down = self.findChild(QtWidgets.QPushButton, "button_down")
        self.button_save = self.findChild(QtWidgets.QPushButton, "button_save")
        self.button_cancel = self.findChild(QtWidgets.QPushButton, "button_cancel")
        card_layout = self.card.layout()

        initial_page = self.tab_widget.widget(0)
        if initial_page is not None:
            direct_children = initial_page.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly)
            if direct_children:
                self.content_container = direct_children[0]
        self.preset_ribbon = PresetRibbon(self.card)
        if card_layout is not None:
            card_layout.setSpacing(0)
            card_layout.removeWidget(self.tab_widget)
            self.tab_widget.hide()
            self.tab_widget.setParent(self)
            card_layout.insertWidget(0, self.preset_ribbon, 0)
            if self.content_container is not None:
                self.content_container.setParent(self.card)
                card_layout.insertWidget(1, self.content_container, 1)
                if self.content_container.layout() is not None:
                    self.content_container.layout().setContentsMargins(0, 0, 0, 0)

        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["유형", "순서", "이름"])
        self.table_widget.horizontalHeader().hide()
        self.table_widget.verticalHeader().hide()
        self.table_widget.setShowGrid(False)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.verticalHeader().setDefaultSectionSize(44)
        self.table_widget.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.table_widget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_widget.setColumnWidth(0, 94)
        self.table_widget.setColumnWidth(1, 56)
        self.table_widget.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)

        for button in (
            self.button_add,
            self.button_delay,
            self.button_note,
            self.button_priority,
            self.button_program,
            self.button_cancel,
            self.button_save,
        ):
            button.setMinimumHeight(42)
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)

        self.move_card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        self.move_card.setMinimumHeight(42)
        self.move_card.setMaximumHeight(16777215)
        for button in (self.button_up, self.button_down):
            button.setMinimumHeight(42)
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)

        self.combo_window = QtWidgets.QComboBox(self)
        self.combo_window.hide()

        self.label_title = QtWidgets.QLabel(self)
        self.label_title.hide()
        self.button_arrowup = QtWidgets.QPushButton(self)
        self.button_arrowup.hide()
        self.button_arrowdown = QtWidgets.QPushButton(self)
        self.button_arrowdown.hide()
        self.program_label = QtWidgets.QLabel(self)
        self.program_label.hide()

    def _connect_signals(self):
        self.button_add.clicked.connect(self.btn_trigger)
        self.button_delay.clicked.connect(self.btn_delay)
        self.button_note.clicked.connect(self.btn_note)
        self.button_priority.clicked.connect(self.btn_toggle_type)
        self.button_program.clicked.connect(self.btn_program)
        self.button_save.clicked.connect(self.btn_save)
        self.button_cancel.clicked.connect(self.reject)
        self.button_up.clicked.connect(lambda: self.btn_move(-1))
        self.button_down.clicked.connect(lambda: self.btn_move(1))
        self.table_widget.customContextMenuRequested.connect(self.open_table_context_menu)
        self.table_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.preset_ribbon.tab_selected.connect(self.on_tab_changed)
        self.preset_ribbon.tab_close_requested.connect(self.close_tab)
        self.preset_ribbon.tab_rename_requested.connect(self.rename_tab)
        self.preset_ribbon.add_requested.connect(self.add_tab)

        shortcut_copy = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+C"), self)
        shortcut_copy.activated.connect(self.copy_selected_step)
        self._shortcuts.append(shortcut_copy)

        shortcut_paste = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+V"), self)
        shortcut_paste.activated.connect(self.paste_from_shortcut)
        self._shortcuts.append(shortcut_paste)

        shortcut_edit = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Return), self)
        shortcut_edit.activated.connect(self.btn_edit)
        self._shortcuts.append(shortcut_edit)

        shortcut_edit_enter = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Enter), self)
        shortcut_edit_enter.activated.connect(self.btn_edit)
        self._shortcuts.append(shortcut_edit_enter)

        shortcut_delete = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self)
        shortcut_delete.activated.connect(self.btn_delete)
        self._shortcuts.append(shortcut_delete)

    def reload_everything(self):
        self.macro_data = self.data_manager.get_macro(self.macro_key)
        self._rebuild_tabs()
        self.load_actions()
        self._load_programs()

    def _rebuild_tabs(self):
        tabs = []
        for preset_id, preset in self.macro_data.get("presets", {}).items():
            tabs.append({"id": preset_id, "name": preset.get("name", preset_id)})
        self.preset_ribbon.set_tabs(tabs, self.current_preset_id, self.macro_data.get("default_preset_id"))

    def _load_programs(self):
        current_program = self.macro_data.get("program")
        self.combo_window.clear()
        self.combo_window.addItem("선택 안 함")
        for name in ["Google Chrome", "Safari", "Firefox", "Designer", "Visual Studio Code", "PyCharm"]:
            self.combo_window.addItem(name)
        if current_program:
            index = self.combo_window.findText(current_program)
            if index >= 0:
                self.combo_window.setCurrentIndex(index)
        self._sync_program_button_label()

    def _sync_program_button_label(self):
        self.button_program.setText("매크로 설정")

    def _current_preset(self):
        return self.data_manager.get_active_preset(self.macro_key, self.current_preset_id)

    def _selected_step(self):
        row = self.table_widget.currentRow()
        preset = self._current_preset()
        if row < 0 or not preset or row >= len(preset.get("steps", [])):
            return None, None, row
        step = preset["steps"][row]
        item = self.macro_data.get("items", {}).get(step.get("item_id")) if step.get("step_type") == "item" else None
        return step, item, row

    def load_actions(self):
        preset = self._current_preset()
        steps = self.data_manager.sort_preset_steps(self.macro_key, self.current_preset_id) if preset else []
        self.table_widget.clearSpans()
        self.table_widget.setRowCount(len(steps))

        for row, step in enumerate(steps):
            if step.get("step_type") == "delay":
                badge_text = "대기"
                name = f"딜레이 {step.get('delay_sec', 0)}초"
                badge_bg = QtGui.QColor("#d1d5db")
                badge_fg = QtGui.QColor("#111827")
            elif step.get("step_type") == "note":
                badge_item = QtWidgets.QTableWidgetItem("메모")
                badge_item.setBackground(QtGui.QColor("#e0e7ff"))
                badge_item.setForeground(QtGui.QBrush(QtGui.QColor("#3730a3")))
                badge_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

                note_item = QtWidgets.QTableWidgetItem(str(step.get("note_text") or "메모"))
                note_item.setForeground(QtGui.QBrush(QtGui.QColor("#334155")))
                note_item.setBackground(QtGui.QColor("#f8fafc"))
                note_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)

                self.table_widget.setItem(row, 0, badge_item)
                self.table_widget.setSpan(row, 1, 1, 2)
                self.table_widget.setItem(row, 1, note_item)
                self.table_widget.setItem(row, 2, QtWidgets.QTableWidgetItem(""))
                self.table_widget.setRowHeight(row, 44)
                continue
            else:
                item = self.macro_data.get("items", {}).get(step.get("item_id"), {})
                is_text = item.get("item_type") == "text"
                badge_text = "텍스트" if is_text else "버튼"
                clicks = int(step.get("click_count", 1) or 1)
                suffix = f"  x{clicks}" if (not is_text and clicks > 1) else ""
                name = f"{item.get('name', step.get('item_id'))}{suffix}"
                if is_text:
                    badge_bg = QtGui.QColor("#86efac")
                    badge_fg = QtGui.QColor("#14532d")
                else:
                    badge_bg = QtGui.QColor("#fee2e2")
                    badge_fg = QtGui.QColor("#991b1b")

            badge_item = QtWidgets.QTableWidgetItem(badge_text)
            badge_item.setBackground(badge_bg)
            badge_item.setForeground(QtGui.QBrush(badge_fg))
            badge_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            order_item = QtWidgets.QTableWidgetItem(str(step.get("order", row + 1)))
            order_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            name_item = QtWidgets.QTableWidgetItem(name)

            self.table_widget.setItem(row, 0, badge_item)
            self.table_widget.setItem(row, 1, order_item)
            self.table_widget.setItem(row, 2, name_item)
            self.table_widget.setRowHeight(row, 44)

        if steps:
            self.table_widget.setCurrentCell(0, 2)

    def btn_trigger(self):
        dialog = TriggerDialog(self.macro_data, self.current_preset_id, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dialog.result_payload:
            return
        self.data_manager.set_macro_trigger(self.macro_key, dialog.result_payload["macro_trigger"])
        for preset_id, trigger in dialog.result_payload["preset_triggers"].items():
            self.data_manager.set_preset_trigger(self.macro_key, preset_id, trigger)
        self.reload_everything()

    def btn_delay(self):
        value, ok = QtWidgets.QInputDialog.getDouble(self, "대기시간 추가", "초", 0.5, 0.1, 999.0, 1)
        if not ok:
            return
        self.data_manager.add_delay_step(self.macro_key, self.current_preset_id, value)
        self.reload_everything()

    def btn_note(self):
        note_text, ok = QtWidgets.QInputDialog.getText(self, "메모 추가", "메모", text="메모")
        if not ok:
            return
        self.data_manager.add_note_step(self.macro_key, self.current_preset_id, note_text)
        self.reload_everything()

    def btn_toggle_type(self):
        step, item, _ = self._selected_step()
        if not step or not item:
            return
        next_type = "text" if item.get("item_type") != "text" else "button"
        self.data_manager.set_item_type(self.macro_key, item["id"], next_type)
        self.reload_everything()

    def btn_edit(self):
        step, item, row = self._selected_step()
        if not step:
            return

        if step.get("step_type") == "delay":
            value, ok = QtWidgets.QInputDialog.getDouble(
                self, "대기 수정", "초", float(step.get("delay_sec", 0.5) or 0.5), 0.1, 999.0, 1
            )
            if ok:
                self.data_manager.update_step(self.macro_key, self.current_preset_id, row, delay_sec=value)
                self.reload_everything()
            return
        if step.get("step_type") == "note":
            note_text, ok = QtWidgets.QInputDialog.getText(
                self,
                "메모 수정",
                "메모",
                text=str(step.get("note_text") or "메모"),
            )
            if ok:
                self.data_manager.update_step(self.macro_key, self.current_preset_id, row, note_text=note_text.strip() or "메모")
                self.reload_everything()
            return

        dialog = ItemEditDialog(item, int(step.get("click_count", 1) or 1), self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, clicks, rect = dialog.get_values()
        self.data_manager.update_item(
            self.macro_key,
            item["id"],
            name=name,
            screen_rect=rect,
            capture_rect=dict(rect),
        )
        if item.get("item_type") == "text":
            self.reload_everything()
            return
        self.data_manager.update_step(self.macro_key, self.current_preset_id, row, click_count=int(clicks or 1))
        self.reload_everything()

    def btn_delete(self):
        step, item, row = self._selected_step()
        if not step:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "삭제 확인",
            "선택 항목을 삭제하시겠습니까?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        preset = self._current_preset()
        preset["steps"].pop(row)
        if item:
            still_used = any(
                s.get("item_id") == item["id"]
                for preset_data in self.macro_data.get("presets", {}).values()
                for s in preset_data.get("steps", [])
            )
            if not still_used:
                self.macro_data.get("items", {}).pop(item["id"], None)
        self.data_manager.sort_preset_steps(self.macro_key, self.current_preset_id)
        self.data_manager.save_data()
        self.reload_everything()

    def btn_move(self, direction: int):
        _, _, row = self._selected_step()
        if row < 0:
            return
        moved = self.data_manager.move_step(self.macro_key, self.current_preset_id, row, direction)
        self.reload_everything()
        if moved:
            new_row = max(0, row + direction)
            if new_row < self.table_widget.rowCount():
                self.table_widget.setCurrentCell(new_row, 2)

    def btn_program(self):
        settings = self.macro_data.setdefault("settings", {})
        dialog = MacroSettingsDialog(settings, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        repeat_count, repeat_delay = dialog.values()
        settings["repeat_count"] = int(repeat_count)
        settings["repeat_delay_sec"] = float(repeat_delay)
        self.data_manager.save_data()
        self.reload_everything()

    def copy_selected_step(self):
        step, item, _ = self._selected_step()
        if not step:
            return
        self._copied_step = {
            "step": copy.deepcopy(step),
            "item": copy.deepcopy(item) if item else None,
        }

    def _build_copied_item_name(self, base_name: str) -> str:
        candidate = f"{base_name} 복사본"
        existing = {
            str(item.get("name") or "")
            for item in self.macro_data.get("items", {}).values()
            if isinstance(item, dict)
        }
        if candidate not in existing:
            return candidate
        index = 2
        while True:
            named = f"{candidate} {index}"
            if named not in existing:
                return named
            index += 1

    def paste_copied_step(self, row_hint: int | None = None):
        if not self._copied_step:
            return
        preset = self._current_preset()
        if not preset:
            return

        steps = self.data_manager.sort_preset_steps(self.macro_key, self.current_preset_id)
        insert_index = len(steps) if row_hint is None or row_hint < 0 else min(row_hint + 1, len(steps))
        new_step = copy.deepcopy(self._copied_step["step"])

        if new_step.get("step_type") == "item":
            source_item = self._copied_step.get("item")
            if not source_item:
                return
            new_item = copy.deepcopy(source_item)
            new_item_id = self.data_manager._next_item_id(self.macro_data)
            new_item["id"] = new_item_id
            new_item["name"] = self._build_copied_item_name(str(new_item.get("name") or "항목"))
            self.macro_data.setdefault("items", {})[new_item_id] = new_item
            new_step["item_id"] = new_item_id

        steps.insert(insert_index, new_step)
        for index, step in enumerate(steps, start=1):
            step["order"] = index
        preset["steps"] = steps
        self.data_manager.sort_preset_steps(self.macro_key, self.current_preset_id)
        self.data_manager.save_data()
        self.reload_everything()
        target_row = min(insert_index, self.table_widget.rowCount() - 1)
        if target_row >= 0:
            self.table_widget.setCurrentCell(target_row, 2)

    def paste_from_shortcut(self):
        row = self.table_widget.currentRow()
        self.paste_copied_step(row if row >= 0 else None)

    def open_table_context_menu(self, position):
        row = self.table_widget.rowAt(position.y())
        menu = QtWidgets.QMenu(self)
        action_edit = menu.addAction("수정")
        action_delete = menu.addAction("삭제")
        menu.addSeparator()
        action_copy = menu.addAction("복사")
        action_paste = menu.addAction("붙여넣기")
        if row >= 0:
            self.table_widget.setCurrentCell(row, 2)
        else:
            action_edit.setEnabled(False)
            action_delete.setEnabled(False)
            action_copy.setEnabled(False)
        action_paste.setEnabled(self._copied_step is not None)
        selected = menu.exec(self.table_widget.viewport().mapToGlobal(position))
        if selected == action_edit:
            self.btn_edit()
        elif selected == action_delete:
            self.btn_delete()
        elif selected == action_copy:
            self.copy_selected_step()
        elif selected == action_paste:
            self.paste_copied_step(row if row >= 0 else None)

    def on_item_double_clicked(self, item):
        row = item.row()
        if row >= 0:
            self.table_widget.setCurrentCell(row, 2)
        self.btn_edit()

    def btn_save(self):
        self.macro_data["program"] = None if self.combo_window.currentText() == "선택 안 함" else self.combo_window.currentText()
        valid, message = self.data_manager.validate_macro_configuration(self.macro_key)
        if not valid:
            QtWidgets.QMessageBox.warning(self, "저장 오류", message)
            return
        self.data_manager.save_data()
        self._committed = True
        self.accept()

    def _restore_snapshot(self):
        if self._committed or self._macro_snapshot is None:
            return
        self.data_manager._data.get("macro_list", {})[self.macro_key] = copy.deepcopy(self._macro_snapshot)
        self.data_manager.save_data()
        self.macro_data = self.data_manager.get_macro(self.macro_key)

    def reject(self):
        self._restore_snapshot()
        super().reject()

    def closeEvent(self, event: QtGui.QCloseEvent):
        if not self._committed:
            self._restore_snapshot()
        super().closeEvent(event)

    def add_tab(self):
        new_preset_id = self.data_manager.add_preset(self.macro_key, self.current_preset_id)
        if not new_preset_id:
            return
        self.current_preset_id = new_preset_id
        self.reload_everything()

    def on_tab_changed(self, preset_id: str):
        if preset_id:
            self.current_preset_id = preset_id
            self._rebuild_tabs()
            self.load_actions()

    def rename_tab(self, preset_id: str):
        if not preset_id:
            return
        preset = self.macro_data.get("presets", {}).get(preset_id)
        current_name = preset.get("name", preset_id)
        new_name, ok = QtWidgets.QInputDialog.getText(self, "탭 이름 수정", "이름", text=current_name)
        if ok:
            self.data_manager.rename_preset(self.macro_key, preset_id, new_name)
            self.reload_everything()

    def close_tab(self, preset_id: str):
        if not preset_id or preset_id == self.macro_data.get("default_preset_id"):
            return
        preset = self.macro_data.get("presets", {}).get(preset_id, {})
        preset_name = preset.get("name", preset_id)
        reply = QtWidgets.QMessageBox.question(
            self,
            "프리셋 삭제",
            f"'{preset_name}' 프리셋을 정말 삭제하시겠습니까?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.data_manager.delete_preset(self.macro_key, preset_id)
        if self.current_preset_id == preset_id:
            self.current_preset_id = self.macro_data.get("default_preset_id")
        self.reload_everything()
