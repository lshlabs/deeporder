# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainsettingWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFrame,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QWidget)

class Ui_ImageWindow(object):
    def setupUi(self, ImageWindow):
        if not ImageWindow.objectName():
            ImageWindow.setObjectName(u"ImageWindow")
        ImageWindow.resize(360, 320)
        font = QFont()
        font.setFamilies([u"Nanum Gothic"])
        ImageWindow.setFont(font)
        ImageWindow.setStyleSheet(u"QMainWindow{\n"
"	background-color: #EFEFEF;\n"
"}")
        ImageWindow.setSizeGripEnabled(False)
        ImageWindow.setModal(True)
        self.frame_main = QFrame(ImageWindow)
        self.frame_main.setObjectName(u"frame_main")
        self.frame_main.setGeometry(QRect(10, 10, 340, 300))
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
        self.button_save.setGeometry(QRect(160, 260, 80, 30))
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
        self.button_cancel.setGeometry(QRect(250, 260, 80, 30))
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
        self.label_tip.setGeometry(QRect(30, 30, 200, 20))
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(9)
        self.label_tip.setFont(font3)
        self.label_tip.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_tip.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.comboBox_run = QComboBox(self.frame_main)
        self.comboBox_run.setObjectName(u"comboBox_run")
        self.comboBox_run.setGeometry(QRect(30, 60, 150, 25))
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(10)
        self.comboBox_run.setFont(font4)
        self.comboBox_run.setStyleSheet(u"QComboBox\n"
"{\n"
"  selection-color: white;\n"
"  selection-background-color: blue;\n"
"}")
        self.lineEdit_width = QLineEdit(self.frame_main)
        self.lineEdit_width.setObjectName(u"lineEdit_width")
        self.lineEdit_width.setGeometry(QRect(70, 110, 80, 30))
        self.lineEdit_width.setFont(font4)
        self.lineEdit_width.setStyleSheet(u"color:black;\n"
"border-radius: 5px;\n"
"padding-left:3px;")
        self.lineEdit_width.setMaxLength(32767)
        self.lineEdit_height = QLineEdit(self.frame_main)
        self.lineEdit_height.setObjectName(u"lineEdit_height")
        self.lineEdit_height.setGeometry(QRect(210, 110, 80, 30))
        self.lineEdit_height.setFont(font4)
        self.lineEdit_height.setStyleSheet(u"color:black;\n"
"border-radius: 5px;\n"
"padding-left:3px;")
        self.lineEdit_height.setMaxLength(32767)
        self.label_width = QLabel(self.frame_main)
        self.label_width.setObjectName(u"label_width")
        self.label_width.setGeometry(QRect(40, 110, 30, 30))
        self.label_width.setFont(font4)
        self.label_width.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_width.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.label_height = QLabel(self.frame_main)
        self.label_height.setObjectName(u"label_height")
        self.label_height.setGeometry(QRect(180, 110, 30, 30))
        self.label_height.setFont(font4)
        self.label_height.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_height.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.retranslateUi(ImageWindow)

        self.button_save.setDefault(False)
        self.button_cancel.setDefault(False)


        QMetaObject.connectSlotsByName(ImageWindow)
    # setupUi

    def retranslateUi(self, ImageWindow):
        ImageWindow.setWindowTitle(QCoreApplication.translate("ImageWindow", u"\uc124\uc815", None))
        self.button_save.setText(QCoreApplication.translate("ImageWindow", u"\uc800\uc7a5", None))
        self.button_cancel.setText(QCoreApplication.translate("ImageWindow", u"\ucde8\uc18c", None))
        self.label_tip.setText(QCoreApplication.translate("ImageWindow", u"\ubaa8\ub2c8\ud130\uc758 \ud574\uc0c1\ub3c4\ub97c \uc120\ud0dd\ud574\uc8fc\uc138\uc694", None))
        self.comboBox_run.setPlaceholderText(QCoreApplication.translate("ImageWindow", u"\uc120\ud0dd\ud558\uc138\uc694", None))
        self.lineEdit_width.setInputMask("")
        self.lineEdit_width.setText("")
        self.lineEdit_width.setPlaceholderText(QCoreApplication.translate("ImageWindow", u"ex) 1920", None))
        self.lineEdit_height.setInputMask("")
        self.lineEdit_height.setText("")
        self.lineEdit_height.setPlaceholderText(QCoreApplication.translate("ImageWindow", u"ex) 1080", None))
        self.label_width.setText(QCoreApplication.translate("ImageWindow", u"\uac00\ub85c :", None))
        self.label_height.setText(QCoreApplication.translate("ImageWindow", u"\uc138\ub85c :", None))
    # retranslateUi

