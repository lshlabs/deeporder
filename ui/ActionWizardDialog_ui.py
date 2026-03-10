# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ActionWizardDialog.ui'
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
    QLabel, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_ActionWizardDialog(object):
    def setupUi(self, ActionWizardDialog):
        if not ActionWizardDialog.objectName():
            ActionWizardDialog.setObjectName(u"ActionWizardDialog")
        ActionWizardDialog.resize(460, 260)
        ActionWizardDialog.setMinimumSize(QSize(460, 260))
        ActionWizardDialog.setMaximumSize(QSize(460, 260))
        ActionWizardDialog.setStyleSheet(u"QDialog { background: #f3f4f6; }\n"
"QFrame#Card { background: white; border: 1px solid #d1d5db; border-radius: 14px; }\n"
"QLabel { color: #111827; font-family: 'Malgun Gothic'; }\n"
"QPushButton {\n"
"  min-height: 36px;\n"
"  padding: 0 14px;\n"
"  border-radius: 10px;\n"
"  border: 1px solid #c7ccd4;\n"
"  background: #ffffff;\n"
"  color: #111827;\n"
"  font: 9pt 'Malgun Gothic';\n"
"}\n"
"QPushButton#PrimaryButton {\n"
"  background: #0ea5e9;\n"
"  border: 1px solid #0284c7;\n"
"  color: white;\n"
"  font-weight: 700;\n"
"}\n"
"QPushButton:disabled { background: #dbeafe; color: #6b7280; border-color: #bfdbfe; }\n"
"QLabel#label_chip { background: #e0f2fe; color: #0369a1; border-radius: 10px; padding: 6px 10px; font: 700 8pt 'Malgun Gothic'; }\n"
"QLabel#label_title { font: 700 13pt 'Malgun Gothic'; }\n"
"QLabel#label_status { font: 9pt 'Malgun Gothic'; }\n"
"QLabel#label_hint { color: #4b5563; font: 8pt 'Malgun Gothic'; }\n"
"")
        self.root_layout = QVBoxLayout(ActionWizardDialog)
        self.root_layout.setObjectName(u"root_layout")
        self.root_layout.setContentsMargins(18, 18, 18, 18)
        self.Card = QFrame(ActionWizardDialog)
        self.Card.setObjectName(u"Card")
        self.card_layout = QVBoxLayout(self.Card)
        self.card_layout.setSpacing(14)
        self.card_layout.setObjectName(u"card_layout")
        self.card_layout.setContentsMargins(22, 20, 22, 18)
        self.label_chip = QLabel(self.Card)
        self.label_chip.setObjectName(u"label_chip")
        self.label_chip.setAlignment(Qt.AlignCenter)
        self.label_chip.setMaximumSize(QSize(120, 16777215))

        self.card_layout.addWidget(self.label_chip)

        self.label_title = QLabel(self.Card)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setWordWrap(True)

        self.card_layout.addWidget(self.label_title)

        self.label_status = QLabel(self.Card)
        self.label_status.setObjectName(u"label_status")
        self.label_status.setWordWrap(True)

        self.card_layout.addWidget(self.label_status)

        self.label_hint = QLabel(self.Card)
        self.label_hint.setObjectName(u"label_hint")
        self.label_hint.setWordWrap(True)

        self.card_layout.addWidget(self.label_hint)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.card_layout.addItem(self.verticalSpacer)

        self.action_row = QHBoxLayout()
        self.action_row.setSpacing(10)
        self.action_row.setObjectName(u"action_row")
        self.action_spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.action_row.addItem(self.action_spacer)

        self.button_cancel = QPushButton(self.Card)
        self.button_cancel.setObjectName(u"button_cancel")

        self.action_row.addWidget(self.button_cancel)

        self.button_start = QPushButton(self.Card)
        self.button_start.setObjectName(u"button_start")

        self.action_row.addWidget(self.button_start)


        self.card_layout.addLayout(self.action_row)


        self.root_layout.addWidget(self.Card)


        self.retranslateUi(ActionWizardDialog)

        QMetaObject.connectSlotsByName(ActionWizardDialog)
    # setupUi

    def retranslateUi(self, ActionWizardDialog):
        ActionWizardDialog.setWindowTitle(QCoreApplication.translate("ActionWizardDialog", u"\uac10\uc2dc\ubaa8\ub4dc \uc9c4\uc785", None))
        self.label_chip.setText(QCoreApplication.translate("ActionWizardDialog", u"MONITOR MODE", None))
        self.label_title.setText("")
        self.label_status.setText("")
        self.label_hint.setText(QCoreApplication.translate("ActionWizardDialog", u"\uc774\ubbf8\uc9c0 \uc5c5\ub85c\ub4dc \uc5c6\uc774 \ubc14\ub85c \ud654\uba74\uc744 \ucea1\ucc98\ud558\uace0, \uc774\uc5b4\uc11c \uc601\uc5ed \uc120\ud0dd\uc73c\ub85c \uc774\ub3d9\ud569\ub2c8\ub2e4.", None))
        self.button_cancel.setText(QCoreApplication.translate("ActionWizardDialog", u"\ucde8\uc18c", None))
        self.button_start.setObjectName(QCoreApplication.translate("ActionWizardDialog", u"PrimaryButton", None))
        self.button_start.setText(QCoreApplication.translate("ActionWizardDialog", u"\uac10\uc2dc\ubaa8\ub4dc \uc2dc\uc791", None))
    # retranslateUi

