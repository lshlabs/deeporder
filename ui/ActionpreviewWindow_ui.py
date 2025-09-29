# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ActionpreviewWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFrame,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QWidget)

class Ui_ActionpreviewWindow(object):
    def setupUi(self, ActionpreviewWindow):
        if not ActionpreviewWindow.objectName():
            ActionpreviewWindow.setObjectName(u"ActionpreviewWindow")
        ActionpreviewWindow.resize(360, 490)
        font = QFont()
        font.setFamilies([u"Arial"])
        ActionpreviewWindow.setFont(font)
        ActionpreviewWindow.setStyleSheet(u"QDialog{\n"
"	background-color: #EFEFEF;\n"
"}")
        ActionpreviewWindow.setSizeGripEnabled(False)
        ActionpreviewWindow.setModal(True)
        self.frame_main = QFrame(ActionpreviewWindow)
        self.frame_main.setObjectName(u"frame_main")
        self.frame_main.setGeometry(QRect(10, 10, 340, 470))
        self.frame_main.setFont(font)
        self.frame_main.setStyleSheet(u"QFrame{\n"
"	background: white;\n"
"	border:1px solid black;\n"
"	border-radius: 5px;\n"
"}\n"
"")
        self.frame_main.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_main.setFrameShadow(QFrame.Shadow.Raised)
        self.button_save = QPushButton(self.frame_main)
        self.button_save.setObjectName(u"button_save")
        self.button_save.setGeometry(QRect(150, 430, 80, 30))
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
        self.button_cancel.setGeometry(QRect(240, 430, 80, 30))
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
        self.label_preview = QLabel(self.frame_main)
        self.label_preview.setObjectName(u"label_preview")
        self.label_preview.setGeometry(QRect(20, 20, 300, 200))
        self.label_preview.setMinimumSize(QSize(0, 0))
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(10)
        font2.setBold(False)
        self.label_preview.setFont(font2)
        self.label_preview.setStyleSheet(u"QLabel {\n"
"	color: black;\n"
"	background-color: #EAEFEF;\n"
"	border: 1px solid black;\n"
"}")
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lineEdit = QLineEdit(self.frame_main)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(20, 230, 300, 30))
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(11)
        self.lineEdit.setFont(font3)
        self.lineEdit.setStyleSheet(u"QLineEdit{\n"
"	color:black;\n"
"	border: 1px solid black;\n"
"	border-radius: 5px; \n"
"	padding-left:10px;\n"
"}")
        self.lineEdit.setMaxLength(32767)
        self.groupBox = QGroupBox(self.frame_main)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(20, 280, 300, 90))
        self.groupBox.setFont(font)
        self.groupBox.setStyleSheet(u"QGroupBox{\n"
"	background: white;\n"
"	border:1px solid black;\n"
"	border-radius: 5px;\n"
"}\n"
"")
        self.lineEdit_Y = QLineEdit(self.groupBox)
        self.lineEdit_Y.setObjectName(u"lineEdit_Y")
        self.lineEdit_Y.setGeometry(QRect(190, 35, 80, 30))
        self.lineEdit_Y.setFont(font3)
        self.lineEdit_Y.setStyleSheet(u"QLineEdit{\n"
"	color:black;\n"
"	border: 1px solid black;\n"
"	border-radius: 5px; \n"
"	padding-left:10px;\n"
"}")
        self.lineEdit_Y.setMaxLength(32767)
        self.lineEdit_X = QLineEdit(self.groupBox)
        self.lineEdit_X.setObjectName(u"lineEdit_X")
        self.lineEdit_X.setGeometry(QRect(50, 35, 80, 30))
        self.lineEdit_X.setFont(font3)
        self.lineEdit_X.setStyleSheet(u"QLineEdit{\n"
"	color:black;\n"
"	border: 1px solid black;\n"
"	border-radius: 5px; \n"
"	padding-left:10px;\n"
"}")
        self.lineEdit_X.setMaxLength(32767)
        self.label_Y = QLabel(self.groupBox)
        self.label_Y.setObjectName(u"label_Y")
        self.label_Y.setGeometry(QRect(150, 35, 30, 30))
        self.label_Y.setFont(font3)
        self.label_Y.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_Y.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_X = QLabel(self.groupBox)
        self.label_X.setObjectName(u"label_X")
        self.label_X.setGeometry(QRect(10, 35, 30, 30))
        self.label_X.setFont(font3)
        self.label_X.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_X.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.checkBox_time = QCheckBox(self.frame_main)
        self.checkBox_time.setObjectName(u"checkBox_time")
        self.checkBox_time.setGeometry(QRect(20, 380, 300, 30))
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(12)
        self.checkBox_time.setFont(font4)

        self.retranslateUi(ActionpreviewWindow)

        self.button_save.setDefault(False)
        self.button_cancel.setDefault(False)


        QMetaObject.connectSlotsByName(ActionpreviewWindow)
    # setupUi

    def retranslateUi(self, ActionpreviewWindow):
        ActionpreviewWindow.setWindowTitle(QCoreApplication.translate("ActionpreviewWindow", u"\ub3d9\uc791 \ud3b8\uc9d1", None))
        self.button_save.setText(QCoreApplication.translate("ActionpreviewWindow", u"\uc800\uc7a5", None))
        self.button_cancel.setText(QCoreApplication.translate("ActionpreviewWindow", u"\ucde8\uc18c", None))
        self.label_preview.setText("")
        self.lineEdit.setInputMask("")
        self.lineEdit.setText("")
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("ActionpreviewWindow", u"\ub3d9\uc791\uc758 \uc774\ub984\uc744 \uc785\ub825\ud558\uc138\uc694", None))
        self.groupBox.setTitle(QCoreApplication.translate("ActionpreviewWindow", u"\ud074\ub9ad \uc601\uc5ed \uc88c\ud45c", None))
        self.lineEdit_Y.setInputMask("")
        self.lineEdit_Y.setText("")
        self.lineEdit_Y.setPlaceholderText("")
        self.lineEdit_X.setInputMask("")
        self.lineEdit_X.setText("")
        self.lineEdit_X.setPlaceholderText("")
        self.label_Y.setText(QCoreApplication.translate("ActionpreviewWindow", u"Y :", None))
        self.label_X.setText(QCoreApplication.translate("ActionpreviewWindow", u"X :", None))
        self.checkBox_time.setText(QCoreApplication.translate("ActionpreviewWindow", u" \uc2dc\uac04 \uc870\uc815 \uad00\ub828 \ub3d9\uc791\uc77c \uacbd\uc6b0 \uccb4\ud06c\ud558\uc138\uc694", None))
    # retranslateUi

