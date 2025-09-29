# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ImageWindow.ui'
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

class Ui_ImageWindow(object):
    def setupUi(self, ImageWindow):
        if not ImageWindow.objectName():
            ImageWindow.setObjectName(u"ImageWindow")
        ImageWindow.resize(360, 550)
        font = QFont()
        font.setFamilies([u"Arial"])
        ImageWindow.setFont(font)
        ImageWindow.setStyleSheet(u"QDialog{\n"
"	background-color: #EFEFEF;\n"
"}")
        ImageWindow.setSizeGripEnabled(False)
        ImageWindow.setModal(True)
        self.frame_main = QFrame(ImageWindow)
        self.frame_main.setObjectName(u"frame_main")
        self.frame_main.setGeometry(QRect(10, 10, 340, 530))
        self.frame_main.setFont(font)
        self.frame_main.setStyleSheet(u"background: white;\n"
"border:1px solid black;\n"
"border-radius: 5px;\n"
"")
        self.frame_main.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_main.setFrameShadow(QFrame.Shadow.Raised)
        self.button_save = QPushButton(self.frame_main)
        self.button_save.setObjectName(u"button_save")
        self.button_save.setGeometry(QRect(160, 490, 80, 30))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_save.sizePolicy().hasHeightForWidth())
        self.button_save.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(10)
        font1.setBold(False)
        font1.setHintingPreference(QFont.PreferVerticalHinting)
        self.button_save.setFont(font1)
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
        self.button_cancel.setGeometry(QRect(250, 490, 80, 30))
        sizePolicy.setHeightForWidth(self.button_cancel.sizePolicy().hasHeightForWidth())
        self.button_cancel.setSizePolicy(sizePolicy)
        self.button_cancel.setFont(font1)
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
        self.label_tip.setGeometry(QRect(20, 280, 300, 20))
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(9)
        self.label_tip.setFont(font2)
        self.label_tip.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview = QLabel(self.frame_main)
        self.label_preview.setObjectName(u"label_preview")
        self.label_preview.setGeometry(QRect(20, 20, 300, 200))
        self.label_preview.setMinimumSize(QSize(0, 0))
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(10)
        font3.setBold(False)
        self.label_preview.setFont(font3)
        self.label_preview.setStyleSheet(u"QLabel {\n"
"	color: black;\n"
"	background-color: #EAEFEF;\n"
"	border: 1px solid black;\n"
"}")
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_reset = QPushButton(self.frame_main)
        self.button_reset.setObjectName(u"button_reset")
        self.button_reset.setGeometry(QRect(280, 180, 30, 30))
        self.button_reset.setMinimumSize(QSize(0, 0))
        self.button_reset.setMaximumSize(QSize(88, 16777215))
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(12)
        self.button_reset.setFont(font4)
        self.button_reset.setStyleSheet(u"background: white;\n"
"color:black;\n"
"border: 1px solid black;\n"
"border-radius: 15px;")
        icon1 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.ViewRefresh))
        self.button_reset.setIcon(icon1)
        self.label_surface2 = QLabel(self.frame_main)
        self.label_surface2.setObjectName(u"label_surface2")
        self.label_surface2.setGeometry(QRect(170, 310, 90, 70))
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(128)
        sizePolicy1.setVerticalStretch(60)
        sizePolicy1.setHeightForWidth(self.label_surface2.sizePolicy().hasHeightForWidth())
        self.label_surface2.setSizePolicy(sizePolicy1)
        self.label_surface2.setBaseSize(QSize(0, 0))
        font5 = QFont()
        font5.setFamilies([u"Arial"])
        font5.setPointSize(20)
        font5.setBold(True)
        self.label_surface2.setFont(font5)
        self.label_surface2.setStyleSheet(u"background-color: darkgray;\n"
"color: white;\n"
"\n"
"border-top: 1px solid black;\n"
"border-bottom: 1px dashed white;\n"
"border-left: 1px dashed white;\n"
"border-right: 1px solid black;\n"
"\n"
"border-top-left-radius :0px;\n"
"border-top-right-radius : 5px;\n"
"border-bottom-left-radius : 0px;\n"
"border-bottom-right-radius : 0px;")
        self.label_surface2.setLineWidth(0)
        self.label_surface2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_surface1 = QLabel(self.frame_main)
        self.label_surface1.setObjectName(u"label_surface1")
        self.label_surface1.setGeometry(QRect(80, 310, 90, 70))
        sizePolicy1.setHeightForWidth(self.label_surface1.sizePolicy().hasHeightForWidth())
        self.label_surface1.setSizePolicy(sizePolicy1)
        self.label_surface1.setBaseSize(QSize(0, 0))
        self.label_surface1.setFont(font5)
        self.label_surface1.setStyleSheet(u"background-color: deepskyblue;\n"
"color: white;\n"
"\n"
"border-top: 1px solid black;\n"
"border-bottom: 1px dashed white;\n"
"border-left: 1px solid black;\n"
"border-right: 1px dashed white;\n"
"\n"
"border-top-left-radius :5px;\n"
"border-top-right-radius : 0px;\n"
"border-bottom-left-radius : 0px;\n"
"border-bottom-right-radius : 0px;")
        self.label_surface1.setLineWidth(0)
        self.label_surface1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_surface3 = QLabel(self.frame_main)
        self.label_surface3.setObjectName(u"label_surface3")
        self.label_surface3.setGeometry(QRect(80, 380, 90, 70))
        sizePolicy1.setHeightForWidth(self.label_surface3.sizePolicy().hasHeightForWidth())
        self.label_surface3.setSizePolicy(sizePolicy1)
        self.label_surface3.setBaseSize(QSize(0, 0))
        self.label_surface3.setFont(font5)
        self.label_surface3.setStyleSheet(u"background-color: darkgray;\n"
"color: white;\n"
"\n"
"border-top: 1px dashed white;\n"
"border-bottom: 1px solid black;\n"
"border-left: 1px solid black;\n"
"border-right: 1px dashed white;\n"
"\n"
"border-top-left-radius :0px;\n"
"border-top-right-radius : 0px;\n"
"border-bottom-left-radius : 5px;\n"
"border-bottom-right-radius : 0px;")
        self.label_surface3.setLineWidth(0)
        self.label_surface3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_surface4 = QLabel(self.frame_main)
        self.label_surface4.setObjectName(u"label_surface4")
        self.label_surface4.setGeometry(QRect(170, 380, 90, 70))
        sizePolicy1.setHeightForWidth(self.label_surface4.sizePolicy().hasHeightForWidth())
        self.label_surface4.setSizePolicy(sizePolicy1)
        self.label_surface4.setBaseSize(QSize(0, 0))
        self.label_surface4.setFont(font5)
        self.label_surface4.setStyleSheet(u"background-color: darkgray;\n"
"color: white;\n"
"\n"
"border-top: 1px dashed white;\n"
"border-bottom: 1px solid black;\n"
"border-left: 1px dashed white;\n"
"border-right: 1px solid black;\n"
"\n"
"border-top-left-radius :0px;\n"
"border-top-right-radius : 0px;\n"
"border-bottom-left-radius : 0px;\n"
"border-bottom-right-radius : 5px;")
        self.label_surface4.setLineWidth(0)
        self.label_surface4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lineEdit = QLineEdit(self.frame_main)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(20, 230, 300, 30))
        font6 = QFont()
        font6.setFamilies([u"Arial"])
        font6.setPointSize(10)
        self.lineEdit.setFont(font6)
        self.lineEdit.setStyleSheet(u"color:black;\n"
"border-radius: 5px; \n"
"padding-left:10px")
        self.lineEdit.setMaxLength(32767)

        self.retranslateUi(ImageWindow)

        self.button_save.setDefault(False)
        self.button_cancel.setDefault(False)


        QMetaObject.connectSlotsByName(ImageWindow)
    # setupUi

    def retranslateUi(self, ImageWindow):
        ImageWindow.setWindowTitle(QCoreApplication.translate("ImageWindow", u"\ub3d9\uc791 \ud3b8\uc9d1", None))
        self.button_save.setText(QCoreApplication.translate("ImageWindow", u"\uc800\uc7a5", None))
        self.button_cancel.setText(QCoreApplication.translate("ImageWindow", u"\ucde8\uc18c", None))
        self.label_tip.setText(QCoreApplication.translate("ImageWindow", u"\ud654\uba74 \ub0b4\uc5d0\uc11c \uc774\ubbf8\uc9c0\uac00 \uc778\uc2dd\ub420 \ubd80\ubd84\uc744 \uc120\ud0dd\ud558\uc138\uc694", None))
        self.label_preview.setText(QCoreApplication.translate("ImageWindow", u"\uc774\uacf3\uc744 \ud074\ub9ad\ud558\uc5ec\n"
"\uc774\ubbf8\uc9c0\ub97c \uc120\ud0dd\ud558\uc138\uc694", None))
        self.button_reset.setText("")
        self.label_surface2.setText(QCoreApplication.translate("ImageWindow", u"2", None))
        self.label_surface1.setText(QCoreApplication.translate("ImageWindow", u"1", None))
        self.label_surface3.setText(QCoreApplication.translate("ImageWindow", u"3", None))
        self.label_surface4.setText(QCoreApplication.translate("ImageWindow", u"4", None))
        self.lineEdit.setInputMask("")
        self.lineEdit.setText("")
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("ImageWindow", u"\ub3d9\uc791\uc758 \uc774\ub984\uc744 \uc785\ub825\ud558\uc138\uc694", None))
    # retranslateUi

