# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'PresetRibbon.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QSizePolicy,
    QWidget)

class Ui_PresetRibbon(object):
    def setupUi(self, preset_ribbon):
        if not preset_ribbon.objectName():
            preset_ribbon.setObjectName(u"preset_ribbon")
        preset_ribbon.setStyleSheet(u"QFrame#preset_ribbon { background: transparent; border: none; }\n"
"QFrame#preset_chip {\n"
"  background: #e4e4e4;\n"
"  border: 1px solid #8e8e8e;\n"
"  border-bottom: none;\n"
"  border-top-left-radius: 7px;\n"
"  border-top-right-radius: 7px;\n"
"  border-bottom-left-radius: 0px;\n"
"  border-bottom-right-radius: 0px;\n"
"}\n"
"QFrame#preset_chip:hover { background: #e9eef5; border-color: #7d8ea3; border-bottom: none; }\n"
"QFrame#preset_chip[selected='true'] {\n"
"  background: #ffffff;\n"
"  border-color: #2f2f2f;\n"
"  border-bottom: none;\n"
"}\n"
"QPushButton#preset_label {\n"
"  border: none;\n"
"  background: transparent;\n"
"  color: #111827;\n"
"  padding: 0 10px;\n"
"  min-height: 22px;\n"
"}\n"
"QPushButton#preset_label:hover { color: #0f172a; }\n"
"QPushButton#preset_close {\n"
"  min-width: 13px; max-width: 13px;\n"
"  min-height: 13px; max-height: 13px;\n"
"  padding: 0px;\n"
"  border-radius: 6px;\n"
"  border: none;\n"
"  background: transparent;\n"
"  color: #6b7280;\n"
"}\n"
"QPushButton#"
                        "preset_close:hover { background: #d7dce3; color: #111827; }\n"
"QPushButton#preset_add {\n"
"  min-width: 28px; max-width: 28px;\n"
"  min-height: 28px; max-height: 28px;\n"
"  padding: 0px;\n"
"  margin: 0px;\n"
"  border-top-left-radius: 7px;\n"
"  border-top-right-radius: 7px;\n"
"  border-bottom-left-radius: 0px;\n"
"  border-bottom-right-radius: 0px;\n"
"  border: 1px solid #8e8e8e;\n"
"  border-bottom: none;\n"
"  background: #e4e4e4;\n"
"  color: #111827;\n"
"}\n"
"QPushButton#preset_add:hover { background: #e9eef5; border-color: #7d8ea3; border-bottom: none; }\n"
"")
        self.ribbon_layout = QHBoxLayout(preset_ribbon)
        self.ribbon_layout.setSpacing(3)
        self.ribbon_layout.setObjectName(u"ribbon_layout")
        self.ribbon_layout.setContentsMargins(0, 0, 0, 0)

        self.retranslateUi(preset_ribbon)

        QMetaObject.connectSlotsByName(preset_ribbon)
    # setupUi

    def retranslateUi(self, preset_ribbon):
        pass
    # retranslateUi

