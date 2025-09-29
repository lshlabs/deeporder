# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Step3Window.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
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
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QPushButton,
    QSizePolicy, QWidget)

class Ui_Step3Window(object):
    def setupUi(self, Step3Window):
        if not Step3Window.objectName():
            Step3Window.setObjectName(u"Step3Window")
        Step3Window.resize(500, 400)
        Step3Window.setStyleSheet(u"QDialog{\n"
"	background-color: white;\n"
"}")
        self.label_preview = QLabel(Step3Window)
        self.label_preview.setObjectName(u"label_preview")
        self.label_preview.setGeometry(QRect(0, 0, 500, 350))
        self.label_preview.setMinimumSize(QSize(0, 0))
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(10)
        font.setBold(False)
        self.label_preview.setFont(font)
        self.label_preview.setStyleSheet(u"QLabel {\n"
"	background-color: #EFEFEF;\n"
"	border: none;\n"
"}")
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_accept = QLabel(Step3Window)
        self.label_accept.setObjectName(u"label_accept")
        self.label_accept.setGeometry(QRect(240, 360, 80, 30))
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(12)
        font1.setBold(False)
        self.label_accept.setFont(font1)
        self.label_accept.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :15px;\n"
"	border-top-right-radius : 15px;\n"
"	border-bottom-left-radius :15px;\n"
"	border-bottom-right-radius : 15px;\n"
"	border: none;\n"
"	background-color: #EFEFEF;\n"
"	color: black;\n"
"}")
        self.label_accept.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_reject = QLabel(Step3Window)
        self.label_reject.setObjectName(u"label_reject")
        self.label_reject.setGeometry(QRect(100, 360, 80, 30))
        self.label_reject.setFont(font1)
        self.label_reject.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :15px;\n"
"	border-top-right-radius : 15px;\n"
"	border-bottom-left-radius :15px;\n"
"	border-bottom-right-radius : 15px;\n"
"	border: none;\n"
"	background-color: #EFEFEF;\n"
"	color: black;\n"
"}")
        self.label_reject.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_save = QPushButton(Step3Window)
        self.button_save.setObjectName(u"button_save")
        self.button_save.setGeometry(QRect(410, 360, 80, 30))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_save.sizePolicy().hasHeightForWidth())
        self.button_save.setSizePolicy(sizePolicy)
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(10)
        font2.setBold(False)
        font2.setHintingPreference(QFont.PreferVerticalHinting)
        self.button_save.setFont(font2)
        self.button_save.setAutoFillBackground(False)
        self.button_save.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.InsertText))
        self.button_save.setIcon(icon)
        self.button_save.setAutoExclusive(False)
        self.button_save.setFlat(False)

        self.retranslateUi(Step3Window)

        self.button_save.setDefault(False)


        QMetaObject.connectSlotsByName(Step3Window)
    # setupUi

    def retranslateUi(self, Step3Window):
        Step3Window.setWindowTitle(QCoreApplication.translate("Step3Window", u"\ubc84\ud2bc\uc704\uce58\uc124\uc815", None))
        self.label_preview.setText("")
        self.label_accept.setText(QCoreApplication.translate("Step3Window", u"\uc811\uc218", None))
        self.label_reject.setText(QCoreApplication.translate("Step3Window", u"\uac70\ubd80", None))
        self.button_save.setText(QCoreApplication.translate("Step3Window", u"\uc800\uc7a5", None))
    # retranslateUi

