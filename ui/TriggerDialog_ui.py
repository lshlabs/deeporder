# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'TriggerDialog.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy,
    QSpacerItem, QVBoxLayout, QWidget)

class Ui_TriggerDialog(object):
    def setupUi(self, TriggerDialog):
        if not TriggerDialog.objectName():
            TriggerDialog.setObjectName(u"TriggerDialog")
        TriggerDialog.resize(760, 760)
        TriggerDialog.setStyleSheet(u"QDialog { background: #f3f4f6; }\n"
"QFrame#DialogCard { background: white; border: 1px solid #d1d5db; border-radius: 14px; }\n"
"QFrame#TriggerCard { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; }\n"
"QPushButton {\n"
"  min-height: 38px;\n"
"  padding: 0 14px;\n"
"  border-radius: 10px;\n"
"  border: 1px solid #d1d5db;\n"
"  background: white;\n"
"  color: #111827;\n"
"  font: 9pt 'Malgun Gothic';\n"
"}\n"
"QPushButton#PrimaryButton { background: #0ea5e9; border-color: #0284c7; color: white; font-weight: 700; }\n"
"QLabel, QCheckBox, QComboBox, QLineEdit { color: #111827; font: 9pt 'Malgun Gothic'; }\n"
"QComboBox, QLineEdit {\n"
"  min-height: 36px;\n"
"  border-radius: 10px;\n"
"  border: 1px solid #d1d5db;\n"
"  background: white;\n"
"  padding-left: 10px;\n"
"}\n"
"QLabel#label_title { font: 700 12pt 'Malgun Gothic'; }\n"
"QLabel#label_subtitle { color: #4b5563; font: 9pt 'Malgun Gothic'; }\n"
"")
        self.root_layout = QVBoxLayout(TriggerDialog)
        self.root_layout.setObjectName(u"root_layout")
        self.root_layout.setContentsMargins(16, 16, 16, 16)
        self.DialogCard = QFrame(TriggerDialog)
        self.DialogCard.setObjectName(u"DialogCard")
        self.card_layout = QVBoxLayout(self.DialogCard)
        self.card_layout.setSpacing(12)
        self.card_layout.setObjectName(u"card_layout")
        self.card_layout.setContentsMargins(16, 16, 16, 16)
        self.label_title = QLabel(self.DialogCard)
        self.label_title.setObjectName(u"label_title")

        self.card_layout.addWidget(self.label_title)

        self.label_subtitle = QLabel(self.DialogCard)
        self.label_subtitle.setObjectName(u"label_subtitle")
        self.label_subtitle.setWordWrap(True)

        self.card_layout.addWidget(self.label_subtitle)

        self.scroll_area = QScrollArea(self.DialogCard)
        self.scroll_area.setObjectName(u"scroll_area")
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_host = QWidget()
        self.scroll_host.setObjectName(u"scroll_host")
        self.scroll_layout = QVBoxLayout(self.scroll_host)
        self.scroll_layout.setSpacing(12)
        self.scroll_layout.setObjectName(u"scroll_layout")
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.scroll_host)

        self.card_layout.addWidget(self.scroll_area)

        self.footer_layout = QHBoxLayout()
        self.footer_layout.setObjectName(u"footer_layout")
        self.footer_spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.footer_layout.addItem(self.footer_spacer)

        self.button_save = QPushButton(self.DialogCard)
        self.button_save.setObjectName(u"button_save")

        self.footer_layout.addWidget(self.button_save)

        self.button_cancel = QPushButton(self.DialogCard)
        self.button_cancel.setObjectName(u"button_cancel")

        self.footer_layout.addWidget(self.button_cancel)


        self.card_layout.addLayout(self.footer_layout)


        self.root_layout.addWidget(self.DialogCard)


        self.retranslateUi(TriggerDialog)

        QMetaObject.connectSlotsByName(TriggerDialog)
    # setupUi

    def retranslateUi(self, TriggerDialog):
        TriggerDialog.setWindowTitle(QCoreApplication.translate("TriggerDialog", u"\ud2b8\ub9ac\uac70 \uc124\uc815", None))
        self.label_title.setText(QCoreApplication.translate("TriggerDialog", u"\ud2b8\ub9ac\uac70 \uc124\uc815", None))
        self.label_subtitle.setText("")
        self.button_save.setObjectName(QCoreApplication.translate("TriggerDialog", u"PrimaryButton", None))
        self.button_save.setText(QCoreApplication.translate("TriggerDialog", u"\uc800\uc7a5", None))
        self.button_cancel.setText(QCoreApplication.translate("TriggerDialog", u"\ucde8\uc18c", None))
    # retranslateUi

