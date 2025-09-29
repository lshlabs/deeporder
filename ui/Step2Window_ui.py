# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Step2Window.ui'
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

class Ui_Step2Window(object):
    def setupUi(self, Step2Window):
        if not Step2Window.objectName():
            Step2Window.setObjectName(u"Step2Window")
        Step2Window.resize(500, 400)
        font = QFont()
        font.setFamilies([u"Arial"])
        Step2Window.setFont(font)
        Step2Window.setStyleSheet(u"QDialog{\n"
"	background-color: white;\n"
"}")
        self.label_preview = QLabel(Step2Window)
        self.label_preview.setObjectName(u"label_preview")
        self.label_preview.setGeometry(QRect(0, 0, 500, 350))
        self.label_preview.setMinimumSize(QSize(0, 0))
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(10)
        font1.setBold(False)
        self.label_preview.setFont(font1)
        self.label_preview.setStyleSheet(u"QLabel {\n"
"	background-color: #EFEFEF;\n"
"	border: none;\n"
"}")
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_plus = QLabel(Step2Window)
        self.label_plus.setObjectName(u"label_plus")
        self.label_plus.setGeometry(QRect(280, 360, 80, 30))
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(12)
        font2.setBold(False)
        self.label_plus.setFont(font2)
        self.label_plus.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :15px;\n"
"	border-top-right-radius : 15px;\n"
"	border-bottom-left-radius :15px;\n"
"	border-bottom-right-radius : 15px;\n"
"	border: none;\n"
"	background-color: #EFEFEF;\n"
"	color: black;\n"
"}")
        self.label_plus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_minus = QLabel(Step2Window)
        self.label_minus.setObjectName(u"label_minus")
        self.label_minus.setGeometry(QRect(20, 360, 80, 30))
        self.label_minus.setFont(font2)
        self.label_minus.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :15px;\n"
"	border-top-right-radius : 15px;\n"
"	border-bottom-left-radius :15px;\n"
"	border-bottom-right-radius : 15px;\n"
"	border: none;\n"
"	background-color: #EFEFEF;\n"
"	color: black;\n"
"}")
        self.label_minus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_time = QLabel(Step2Window)
        self.label_time.setObjectName(u"label_time")
        self.label_time.setGeometry(QRect(150, 360, 80, 30))
        self.label_time.setFont(font2)
        self.label_time.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :15px;\n"
"	border-top-right-radius : 15px;\n"
"	border-bottom-left-radius :15px;\n"
"	border-bottom-right-radius : 15px;\n"
"	border: none;\n"
"	background-color: #EFEFEF;\n"
"	color: black;\n"
"}")
        self.label_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_save = QPushButton(Step2Window)
        self.button_save.setObjectName(u"button_save")
        self.button_save.setGeometry(QRect(410, 360, 80, 30))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_save.sizePolicy().hasHeightForWidth())
        self.button_save.setSizePolicy(sizePolicy)
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(10)
        font3.setBold(False)
        font3.setHintingPreference(QFont.PreferVerticalHinting)
        self.button_save.setFont(font3)
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

        self.retranslateUi(Step2Window)

        self.button_save.setDefault(False)


        QMetaObject.connectSlotsByName(Step2Window)
    # setupUi

    def retranslateUi(self, Step2Window):
        Step2Window.setWindowTitle(QCoreApplication.translate("Step2Window", u"\ubc84\ud2bc\uc704\uce58\uc124\uc815", None))
        self.label_preview.setText("")
        self.label_plus.setText(QCoreApplication.translate("Step2Window", u"+ \ubc84\ud2bc", None))
        self.label_minus.setText(QCoreApplication.translate("Step2Window", u"- \ubc84\ud2bc", None))
        self.label_time.setText(QCoreApplication.translate("Step2Window", u"\uc608\uc0c1\uc2dc\uac04", None))
        self.button_save.setText(QCoreApplication.translate("Step2Window", u"\uc800\uc7a5", None))
    # retranslateUi

