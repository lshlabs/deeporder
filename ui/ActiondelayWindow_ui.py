# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ActiondelayWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QWidget)

class Ui_ActiondelayWindow(object):
    def setupUi(self, ActiondelayWindow):
        if not ActiondelayWindow.objectName():
            ActiondelayWindow.setObjectName(u"ActiondelayWindow")
        ActiondelayWindow.resize(360, 220)
        font = QFont()
        font.setFamilies([u"Nanum Gothic"])
        ActiondelayWindow.setFont(font)
        ActiondelayWindow.setStyleSheet(u"QMainWindow{\n"
"	background-color: #EFEFEF;\n"
"}")
        ActiondelayWindow.setSizeGripEnabled(False)
        ActiondelayWindow.setModal(True)
        self.frame_main = QFrame(ActiondelayWindow)
        self.frame_main.setObjectName(u"frame_main")
        self.frame_main.setGeometry(QRect(10, 10, 340, 200))
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        self.frame_main.setFont(font1)
        self.frame_main.setStyleSheet(u"background: white;\n"
"border:1px solid black;\n"
"border-radius: 5px;\n"
"")
        self.frame_main.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_main.setFrameShadow(QFrame.Shadow.Raised)
        self.button_save = QPushButton(self.frame_main)
        self.button_save.setObjectName(u"button_save")
        self.button_save.setGeometry(QRect(160, 160, 80, 30))
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
        self.button_cancel = QPushButton(self.frame_main)
        self.button_cancel.setObjectName(u"button_cancel")
        self.button_cancel.setGeometry(QRect(250, 160, 80, 30))
        sizePolicy.setHeightForWidth(self.button_cancel.sizePolicy().hasHeightForWidth())
        self.button_cancel.setSizePolicy(sizePolicy)
        self.button_cancel.setFont(font2)
        self.button_cancel.setAutoFillBackground(False)
        self.button_cancel.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_cancel.setIcon(icon)
        self.button_cancel.setAutoExclusive(False)
        self.button_cancel.setFlat(False)
        self.label_tip = QLabel(self.frame_main)
        self.label_tip.setObjectName(u"label_tip")
        self.label_tip.setGeometry(QRect(20, 20, 300, 20))
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(9)
        self.label_tip.setFont(font3)
        self.label_tip.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label1 = QLabel(self.frame_main)
        self.label1.setObjectName(u"label1")
        self.label1.setGeometry(QRect(50, 80, 70, 30))
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(10)
        self.label1.setFont(font4)
        self.label1.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label1.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.lineEdit_delay = QLineEdit(self.frame_main)
        self.lineEdit_delay.setObjectName(u"lineEdit_delay")
        self.lineEdit_delay.setGeometry(QRect(130, 80, 111, 30))
        self.lineEdit_delay.setFont(font4)
        self.lineEdit_delay.setStyleSheet(u"color:black;\n"
"border-radius: 5px;\n"
"padding-left: 3px;")
        self.lineEdit_delay.setMaxLength(32767)
        self.label2 = QLabel(self.frame_main)
        self.label2.setObjectName(u"label2")
        self.label2.setGeometry(QRect(250, 80, 30, 30))
        self.label2.setFont(font4)
        self.label2.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label2.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.retranslateUi(ActiondelayWindow)

        self.button_save.setDefault(False)
        self.button_cancel.setDefault(False)


        QMetaObject.connectSlotsByName(ActiondelayWindow)
    # setupUi

    def retranslateUi(self, ActiondelayWindow):
        ActiondelayWindow.setWindowTitle(QCoreApplication.translate("ActiondelayWindow", u"\ub300\uae30\uc2dc\uac04 \ucd94\uac00", None))
        self.button_save.setText(QCoreApplication.translate("ActiondelayWindow", u"\uc800\uc7a5", None))
        self.button_cancel.setText(QCoreApplication.translate("ActiondelayWindow", u"\ucde8\uc18c", None))
        self.label_tip.setText(QCoreApplication.translate("ActiondelayWindow", u"\ub2e4\uc74c \ub3d9\uc791\uae4c\uc9c0 \ub300\uae30\ud560 \uc2dc\uac04\uc744 \uc124\uc815\ud558\uc138\uc694", None))
        self.label1.setText(QCoreApplication.translate("ActiondelayWindow", u"\ub300\uae30 \uc2dc\uac04 :", None))
        self.lineEdit_delay.setInputMask("")
        self.lineEdit_delay.setText("")
        self.lineEdit_delay.setPlaceholderText(QCoreApplication.translate("ActiondelayWindow", u"ex) 0.5", None))
        self.label2.setText(QCoreApplication.translate("ActiondelayWindow", u"\ucd08", None))
    # retranslateUi

