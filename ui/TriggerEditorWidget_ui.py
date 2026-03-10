# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'TriggerEditorWidget.ui'
##
## Created by: Qt User Interface Compiler version 6.10.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_TriggerEditorWidget(object):
    def setupUi(self, TriggerEditorWidget):
        if not TriggerEditorWidget.objectName():
            TriggerEditorWidget.setObjectName(u"TriggerEditorWidget")
        TriggerEditorWidget.setStyleSheet(u"QLabel#label_title { font: 700 10pt 'Malgun Gothic'; color: #111827; }\n"
"QLabel#guide_label { color: #4b5563; font: 8.5pt 'Malgun Gothic'; padding-left: 2px; }\n"
"QLabel#region_label { color: #4b5563; background: white; border: 1px solid #d1d5db; border-radius: 10px; padding: 0 10px; }\n"
"QLabel#color_chip { background: #f3f4f6; color: #6b7280; border: 1px solid #d1d5db; border-radius: 8px; }\n"
"")
        self.root_layout = QVBoxLayout(TriggerEditorWidget)
        self.root_layout.setSpacing(10)
        self.root_layout.setObjectName(u"root_layout")
        self.root_layout.setContentsMargins(16, 16, 16, 16)
        self.header_row = QHBoxLayout()
        self.header_row.setSpacing(10)
        self.header_row.setObjectName(u"header_row")
        self.label_title = QLabel(TriggerEditorWidget)
        self.label_title.setObjectName(u"label_title")

        self.header_row.addWidget(self.label_title)

        self.check_enabled = QCheckBox(TriggerEditorWidget)
        self.check_enabled.setObjectName(u"check_enabled")

        self.header_row.addWidget(self.check_enabled)


        self.root_layout.addLayout(self.header_row)

        self.row_type = QWidget(TriggerEditorWidget)
        self.row_type.setObjectName(u"row_type")
        self.type_row = QHBoxLayout(self.row_type)
        self.type_row.setSpacing(10)
        self.type_row.setObjectName(u"type_row")
        self.type_row.setContentsMargins(0, 0, 0, 0)
        self.label_type = QLabel(self.row_type)
        self.label_type.setObjectName(u"label_type")
        self.label_type.setMinimumSize(QSize(46, 0))

        self.type_row.addWidget(self.label_type)

        self.combo_type = QComboBox(self.row_type)
        self.combo_type.setObjectName(u"combo_type")

        self.type_row.addWidget(self.combo_type)


        self.root_layout.addWidget(self.row_type)

        self.row_text_item = QWidget(TriggerEditorWidget)
        self.row_text_item.setObjectName(u"row_text_item")
        self.text_item_row = QHBoxLayout(self.row_text_item)
        self.text_item_row.setSpacing(10)
        self.text_item_row.setObjectName(u"text_item_row")
        self.text_item_row.setContentsMargins(0, 0, 0, 0)
        self.label_text_item = QLabel(self.row_text_item)
        self.label_text_item.setObjectName(u"label_text_item")
        self.label_text_item.setMinimumSize(QSize(46, 0))

        self.text_item_row.addWidget(self.label_text_item)

        self.combo_text_item = QComboBox(self.row_text_item)
        self.combo_text_item.setObjectName(u"combo_text_item")

        self.text_item_row.addWidget(self.combo_text_item)


        self.root_layout.addWidget(self.row_text_item)

        self.guide_label = QLabel(TriggerEditorWidget)
        self.guide_label.setObjectName(u"guide_label")
        self.guide_label.setWordWrap(True)

        self.root_layout.addWidget(self.guide_label)

        self.row_region = QWidget(TriggerEditorWidget)
        self.row_region.setObjectName(u"row_region")
        self.region_row = QHBoxLayout(self.row_region)
        self.region_row.setSpacing(10)
        self.region_row.setObjectName(u"region_row")
        self.region_row.setContentsMargins(0, 0, 0, 0)
        self.button_region = QPushButton(self.row_region)
        self.button_region.setObjectName(u"button_region")

        self.region_row.addWidget(self.button_region)

        self.region_label = QLabel(self.row_region)
        self.region_label.setObjectName(u"region_label")
        self.region_label.setMinimumSize(QSize(0, 36))
        self.region_label.setWordWrap(True)

        self.region_row.addWidget(self.region_label)


        self.root_layout.addWidget(self.row_region)

        self.row_color = QWidget(TriggerEditorWidget)
        self.row_color.setObjectName(u"row_color")
        self.color_row = QHBoxLayout(self.row_color)
        self.color_row.setSpacing(10)
        self.color_row.setObjectName(u"color_row")
        self.color_row.setContentsMargins(0, 0, 0, 0)
        self.button_color = QPushButton(self.row_color)
        self.button_color.setObjectName(u"button_color")

        self.color_row.addWidget(self.button_color)

        self.color_chip = QLabel(self.row_color)
        self.color_chip.setObjectName(u"color_chip")
        self.color_chip.setMinimumSize(QSize(92, 32))
        self.color_chip.setMaximumSize(QSize(92, 32))
        self.color_chip.setAlignment(Qt.AlignCenter)

        self.color_row.addWidget(self.color_chip)

        self.button_sample = QPushButton(self.row_color)
        self.button_sample.setObjectName(u"button_sample")

        self.color_row.addWidget(self.button_sample)


        self.root_layout.addWidget(self.row_color)

        self.row_text = QWidget(TriggerEditorWidget)
        self.row_text.setObjectName(u"row_text")
        self.text_row = QHBoxLayout(self.row_text)
        self.text_row.setSpacing(10)
        self.text_row.setObjectName(u"text_row")
        self.text_row.setContentsMargins(0, 0, 0, 0)
        self.label_text = QLabel(self.row_text)
        self.label_text.setObjectName(u"label_text")
        self.label_text.setMinimumSize(QSize(46, 0))

        self.text_row.addWidget(self.label_text)

        self.line_text = QLineEdit(self.row_text)
        self.line_text.setObjectName(u"line_text")

        self.text_row.addWidget(self.line_text)


        self.root_layout.addWidget(self.row_text)


        self.retranslateUi(TriggerEditorWidget)

        QMetaObject.connectSlotsByName(TriggerEditorWidget)
    # setupUi

    def retranslateUi(self, TriggerEditorWidget):
        self.label_title.setText(QCoreApplication.translate("TriggerEditorWidget", u"\ud2b8\ub9ac\uac70", None))
        self.check_enabled.setText(QCoreApplication.translate("TriggerEditorWidget", u"\ud65c\uc131\ud654", None))
        self.label_type.setText(QCoreApplication.translate("TriggerEditorWidget", u"\ud0c0\uc785", None))
        self.label_text_item.setText(QCoreApplication.translate("TriggerEditorWidget", u"\uc694\uc18c", None))
        self.guide_label.setText("")
        self.button_region.setText(QCoreApplication.translate("TriggerEditorWidget", u"\uc601\uc5ed \uc9c0\uc815", None))
        self.region_label.setText(QCoreApplication.translate("TriggerEditorWidget", u"\uc120\ud0dd\ub41c \uc601\uc5ed \uc5c6\uc74c", None))
        self.button_color.setText(QCoreApplication.translate("TriggerEditorWidget", u"\uc0c9\uc0c1 \uace0\ub974\uae30", None))
        self.color_chip.setText(QCoreApplication.translate("TriggerEditorWidget", u"\ubbf8\uc120\ud0dd", None))
        self.button_sample.setText(QCoreApplication.translate("TriggerEditorWidget", u"\uc911\uc559\uc0c9 \ucd94\ucd9c", None))
        self.label_text.setText(QCoreApplication.translate("TriggerEditorWidget", u"\ud14d\uc2a4\ud2b8", None))
        self.line_text.setPlaceholderText(QCoreApplication.translate("TriggerEditorWidget", u"\uc608: 15\ubd84", None))
        pass
    # retranslateUi

