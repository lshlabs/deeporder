# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MacroSettingsDialog.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QDoubleSpinBox, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QSpacerItem,
    QSpinBox, QVBoxLayout, QWidget)

class Ui_MacroSettingsDialog(object):
    def setupUi(self, MacroSettingsDialog):
        if not MacroSettingsDialog.objectName():
            MacroSettingsDialog.setObjectName(u"MacroSettingsDialog")
        MacroSettingsDialog.resize(320, 180)
        MacroSettingsDialog.setModal(True)
        self.root_layout = QVBoxLayout(MacroSettingsDialog)
        self.root_layout.setSpacing(12)
        self.root_layout.setObjectName(u"root_layout")
        self.root_layout.setContentsMargins(16, 16, 16, 16)
        self.repeat_row = QHBoxLayout()
        self.repeat_row.setObjectName(u"repeat_row")
        self.label_repeat = QLabel(MacroSettingsDialog)
        self.label_repeat.setObjectName(u"label_repeat")
        self.label_repeat.setMinimumSize(QSize(80, 0))

        self.repeat_row.addWidget(self.label_repeat)

        self.spin_repeat = QSpinBox(MacroSettingsDialog)
        self.spin_repeat.setObjectName(u"spin_repeat")
        self.spin_repeat.setMinimum(0)
        self.spin_repeat.setMaximum(9999)

        self.repeat_row.addWidget(self.spin_repeat)

        self.label_repeat_hint = QLabel(MacroSettingsDialog)
        self.label_repeat_hint.setObjectName(u"label_repeat_hint")
        self.label_repeat_hint.setStyleSheet(u"color:#4b5563;")

        self.repeat_row.addWidget(self.label_repeat_hint)


        self.root_layout.addLayout(self.repeat_row)

        self.delay_row = QHBoxLayout()
        self.delay_row.setObjectName(u"delay_row")
        self.label_delay = QLabel(MacroSettingsDialog)
        self.label_delay.setObjectName(u"label_delay")
        self.label_delay.setMinimumSize(QSize(80, 0))

        self.delay_row.addWidget(self.label_delay)

        self.spin_delay = QDoubleSpinBox(MacroSettingsDialog)
        self.spin_delay.setObjectName(u"spin_delay")
        self.spin_delay.setDecimals(1)
        self.spin_delay.setMinimum(0.000000000000000)
        self.spin_delay.setMaximum(9999.000000000000000)
        self.spin_delay.setSingleStep(0.100000000000000)

        self.delay_row.addWidget(self.spin_delay)

        self.label_delay_suffix = QLabel(MacroSettingsDialog)
        self.label_delay_suffix.setObjectName(u"label_delay_suffix")

        self.delay_row.addWidget(self.label_delay_suffix)

        self.delay_spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.delay_row.addItem(self.delay_spacer)


        self.root_layout.addLayout(self.delay_row)

        self.footer_row = QHBoxLayout()
        self.footer_row.setObjectName(u"footer_row")
        self.footer_spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.footer_row.addItem(self.footer_spacer)

        self.button_save = QPushButton(MacroSettingsDialog)
        self.button_save.setObjectName(u"button_save")

        self.footer_row.addWidget(self.button_save)

        self.button_cancel = QPushButton(MacroSettingsDialog)
        self.button_cancel.setObjectName(u"button_cancel")

        self.footer_row.addWidget(self.button_cancel)


        self.root_layout.addLayout(self.footer_row)


        self.retranslateUi(MacroSettingsDialog)

        QMetaObject.connectSlotsByName(MacroSettingsDialog)
    # setupUi

    def retranslateUi(self, MacroSettingsDialog):
        MacroSettingsDialog.setWindowTitle(QCoreApplication.translate("MacroSettingsDialog", u"\ub9e4\ud06c\ub85c \uc124\uc815", None))
        self.label_repeat.setText(QCoreApplication.translate("MacroSettingsDialog", u"\uc2e4\ud589 \ud69f\uc218", None))
        self.label_repeat_hint.setText(QCoreApplication.translate("MacroSettingsDialog", u"0 = \ubb34\ud55c \ubc18\ubcf5", None))
        self.label_delay.setText(QCoreApplication.translate("MacroSettingsDialog", u"\ubc18\ubcf5 \ub51c\ub808\uc774", None))
        self.label_delay_suffix.setText(QCoreApplication.translate("MacroSettingsDialog", u"\ucd08", None))
        self.button_save.setText(QCoreApplication.translate("MacroSettingsDialog", u"\uc800\uc7a5", None))
        self.button_cancel.setText(QCoreApplication.translate("MacroSettingsDialog", u"\ucde8\uc18c", None))
    # retranslateUi

