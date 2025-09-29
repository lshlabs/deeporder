# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ActionWizardWindow.ui'
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
    QPushButton, QSizePolicy, QStackedWidget, QWidget)

class Ui_ActionWizardWindow(object):
    def setupUi(self, ActionWizardWindow):
        if not ActionWizardWindow.objectName():
            ActionWizardWindow.setObjectName(u"ActionWizardWindow")
        ActionWizardWindow.resize(500, 470)
        ActionWizardWindow.setMaximumSize(QSize(500, 470))
        font = QFont()
        font.setFamilies([u"Arial"])
        ActionWizardWindow.setFont(font)
        ActionWizardWindow.setAutoFillBackground(False)
        ActionWizardWindow.setStyleSheet(u"QDialog{\n"
"	background-color: #EFEFEF;\n"
"}")
        self.mainwindow = QWidget(ActionWizardWindow)
        self.mainwindow.setObjectName(u"mainwindow")
        self.mainwindow.setGeometry(QRect(0, 0, 500, 470))
        font1 = QFont()
        font1.setFamilies([u".AppleSystemUIFont"])
        font1.setBold(False)
        self.mainwindow.setFont(font1)
        self.stackedWidget = QStackedWidget(self.mainwindow)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setGeometry(QRect(10, 10, 480, 450))
        self.stackedWidget.setFont(font)
        self.stackedWidget.setStyleSheet(u"")
        self.stackedWidget.setLineWidth(0)
        self.page1 = QWidget()
        self.page1.setObjectName(u"page1")
        self.page1.setStyleSheet(u"")
        self.frame1 = QFrame(self.page1)
        self.frame1.setObjectName(u"frame1")
        self.frame1.setGeometry(QRect(0, 0, 480, 450))
        self.frame1.setStyleSheet(u"QFrame {\n"
"	background: white;\n"
"	border:1px solid black;\n"
"	border-radius: 5px;\n"
"}")
        self.frame1.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame1.setFrameShadow(QFrame.Shadow.Raised)
        self.button_next1 = QPushButton(self.frame1)
        self.button_next1.setObjectName(u"button_next1")
        self.button_next1.setGeometry(QRect(390, 410, 80, 30))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_next1.sizePolicy().hasHeightForWidth())
        self.button_next1.setSizePolicy(sizePolicy)
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(10)
        font2.setBold(False)
        font2.setHintingPreference(QFont.PreferVerticalHinting)
        self.button_next1.setFont(font2)
        self.button_next1.setAutoFillBackground(False)
        self.button_next1.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: darkgray;\n"
"    padding: 2px 4px;\n"
"}")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.InsertText))
        self.button_next1.setIcon(icon)
        self.button_next1.setAutoExclusive(False)
        self.button_next1.setFlat(False)
        self.label_title1 = QLabel(self.frame1)
        self.label_title1.setObjectName(u"label_title1")
        self.label_title1.setGeometry(QRect(1, 1, 478, 60))
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(128)
        sizePolicy1.setVerticalStretch(60)
        sizePolicy1.setHeightForWidth(self.label_title1.sizePolicy().hasHeightForWidth())
        self.label_title1.setSizePolicy(sizePolicy1)
        self.label_title1.setBaseSize(QSize(0, 0))
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(20)
        font3.setBold(True)
        self.label_title1.setFont(font3)
        self.label_title1.setStyleSheet(u"QLabel {\n"
"	background-color: deepskyblue;\n"
"	color: white;\n"
"	border: none;\n"
"	border-top-left-radius :5px;\n"
"	border-top-right-radius : 5px;\n"
"	border-bottom-left-radius : 0px;\n"
"	border-bottom-right-radius : 0px;\n"
"}")
        self.label_title1.setLineWidth(0)
        self.label_title1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview1 = QLabel(self.frame1)
        self.label_preview1.setObjectName(u"label_preview1")
        self.label_preview1.setGeometry(QRect(65, 130, 350, 250))
        self.label_preview1.setMinimumSize(QSize(0, 0))
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(12)
        font4.setBold(False)
        self.label_preview1.setFont(font4)
        self.label_preview1.setStyleSheet(u"QLabel {\n"
"	color: black;\n"
"	background-color: #EFEFEF;\n"
"	border-radius: 0px;\n"
"	border: 1px solid #8f8f91;\n"
"}")
        self.label_preview1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_step1 = QLabel(self.frame1)
        self.label_step1.setObjectName(u"label_step1")
        self.label_step1.setGeometry(QRect(60, 80, 80, 30))
        font5 = QFont()
        font5.setFamilies([u"Arial"])
        font5.setPointSize(15)
        font5.setBold(True)
        self.label_step1.setFont(font5)
        self.label_step1.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :15px;\n"
"	border-top-right-radius : 0px;\n"
"	border-bottom-left-radius :15px;\n"
"	border-bottom-right-radius : 0px;\n"
"	border: none;\n"
"	background-color: deepskyblue;\n"
"	color: white;\n"
"}")
        self.label_step1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_step1_text = QLabel(self.frame1)
        self.label_step1_text.setObjectName(u"label_step1_text")
        self.label_step1_text.setGeometry(QRect(140, 80, 280, 30))
        self.label_step1_text.setFont(font4)
        self.label_step1_text.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :0px;\n"
"	border-top-right-radius : 15px;\n"
"	border-bottom-left-radius :0px;\n"
"	border-bottom-right-radius : 15px;\n"
"	border: none;\n"
"	background-color: #EFEFEF;\n"
"	color: black;\n"
"}")
        self.label_step1_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_cancel = QPushButton(self.frame1)
        self.button_cancel.setObjectName(u"button_cancel")
        self.button_cancel.setGeometry(QRect(300, 410, 80, 30))
        sizePolicy.setHeightForWidth(self.button_cancel.sizePolicy().hasHeightForWidth())
        self.button_cancel.setSizePolicy(sizePolicy)
        self.button_cancel.setFont(font2)
        self.button_cancel.setAutoFillBackground(False)
        self.button_cancel.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #f0f0f0;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_cancel.setIcon(icon)
        self.button_cancel.setCheckable(False)
        self.button_cancel.setAutoExclusive(False)
        self.button_cancel.setFlat(False)
        self.stackedWidget.addWidget(self.page1)
        self.page2 = QWidget()
        self.page2.setObjectName(u"page2")
        self.frame2 = QFrame(self.page2)
        self.frame2.setObjectName(u"frame2")
        self.frame2.setGeometry(QRect(0, 0, 480, 450))
        self.frame2.setStyleSheet(u"QFrame {\n"
"	background: white;\n"
"	border:1px solid black;\n"
"	border-radius: 5px;\n"
"}")
        self.frame2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame2.setFrameShadow(QFrame.Shadow.Raised)
        self.button_next2 = QPushButton(self.frame2)
        self.button_next2.setObjectName(u"button_next2")
        self.button_next2.setGeometry(QRect(390, 410, 80, 30))
        sizePolicy.setHeightForWidth(self.button_next2.sizePolicy().hasHeightForWidth())
        self.button_next2.setSizePolicy(sizePolicy)
        self.button_next2.setFont(font2)
        self.button_next2.setAutoFillBackground(False)
        self.button_next2.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: darkgray;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_next2.setIcon(icon)
        self.button_next2.setAutoExclusive(False)
        self.button_next2.setFlat(False)
        self.label_title2 = QLabel(self.frame2)
        self.label_title2.setObjectName(u"label_title2")
        self.label_title2.setGeometry(QRect(1, 1, 478, 60))
        sizePolicy1.setHeightForWidth(self.label_title2.sizePolicy().hasHeightForWidth())
        self.label_title2.setSizePolicy(sizePolicy1)
        self.label_title2.setBaseSize(QSize(0, 0))
        self.label_title2.setFont(font3)
        self.label_title2.setStyleSheet(u"QLabel {\n"
"	background-color: deepskyblue;\n"
"	color: white;\n"
"	border: none;\n"
"	border-top-left-radius :5px;\n"
"	border-top-right-radius : 5px;\n"
"	border-bottom-left-radius : 0px;\n"
"	border-bottom-right-radius : 0px;\n"
"}")
        self.label_title2.setLineWidth(0)
        self.label_title2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_prev1 = QPushButton(self.frame2)
        self.button_prev1.setObjectName(u"button_prev1")
        self.button_prev1.setGeometry(QRect(300, 410, 80, 30))
        sizePolicy.setHeightForWidth(self.button_prev1.sizePolicy().hasHeightForWidth())
        self.button_prev1.setSizePolicy(sizePolicy)
        self.button_prev1.setFont(font2)
        self.button_prev1.setAutoFillBackground(False)
        self.button_prev1.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #f0f0f0;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_prev1.setIcon(icon)
        self.button_prev1.setCheckable(False)
        self.button_prev1.setAutoExclusive(False)
        self.button_prev1.setFlat(False)
        self.label_preview2 = QLabel(self.frame2)
        self.label_preview2.setObjectName(u"label_preview2")
        self.label_preview2.setGeometry(QRect(70, 130, 340, 240))
        self.label_preview2.setMinimumSize(QSize(0, 0))
        font6 = QFont()
        font6.setFamilies([u"Arial"])
        font6.setPointSize(10)
        font6.setBold(False)
        self.label_preview2.setFont(font6)
        self.label_preview2.setStyleSheet(u"QLabel {\n"
"	color: black;\n"
"	background-color: #EFEFEF;\n"
"	border-radius: 0px;\n"
"	border: 1px solid #8f8f91;\n"
"}")
        self.label_preview2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_step2 = QLabel(self.frame2)
        self.label_step2.setObjectName(u"label_step2")
        self.label_step2.setGeometry(QRect(80, 80, 80, 30))
        self.label_step2.setFont(font5)
        self.label_step2.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :15px;\n"
"	border-top-right-radius : 0px;\n"
"	border-bottom-left-radius :15px;\n"
"	border-bottom-right-radius : 0px;\n"
"	border: none;\n"
"	background-color: deepskyblue;\n"
"	color: white;\n"
"}")
        self.label_step2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_step2_text = QLabel(self.frame2)
        self.label_step2_text.setObjectName(u"label_step2_text")
        self.label_step2_text.setGeometry(QRect(160, 80, 240, 30))
        self.label_step2_text.setFont(font4)
        self.label_step2_text.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :0px;\n"
"	border-top-right-radius : 15px;\n"
"	border-bottom-left-radius :0px;\n"
"	border-bottom-right-radius : 15px;\n"
"	border: none;\n"
"	background-color: #EFEFEF;\n"
"	color: black;\n"
"}")
        self.label_step2_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_tip1 = QLabel(self.frame2)
        self.label_tip1.setObjectName(u"label_tip1")
        self.label_tip1.setGeometry(QRect(70, 370, 340, 20))
        font7 = QFont()
        font7.setFamilies([u"Arial"])
        font7.setPointSize(11)
        self.label_tip1.setFont(font7)
        self.label_tip1.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_tip1.setAlignment(Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignHCenter)
        self.stackedWidget.addWidget(self.page2)
        self.page3 = QWidget()
        self.page3.setObjectName(u"page3")
        self.frame3 = QFrame(self.page3)
        self.frame3.setObjectName(u"frame3")
        self.frame3.setGeometry(QRect(0, 0, 480, 450))
        self.frame3.setStyleSheet(u"QFrame {\n"
"	background: white;\n"
"	border:1px solid black;\n"
"	border-radius: 5px;\n"
"}")
        self.frame3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame3.setFrameShadow(QFrame.Shadow.Raised)
        self.button_save = QPushButton(self.frame3)
        self.button_save.setObjectName(u"button_save")
        self.button_save.setGeometry(QRect(390, 410, 80, 30))
        sizePolicy.setHeightForWidth(self.button_save.sizePolicy().hasHeightForWidth())
        self.button_save.setSizePolicy(sizePolicy)
        self.button_save.setFont(font2)
        self.button_save.setAutoFillBackground(False)
        self.button_save.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: darkgray;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_save.setIcon(icon)
        self.button_save.setAutoExclusive(False)
        self.button_save.setFlat(False)
        self.label_title3 = QLabel(self.frame3)
        self.label_title3.setObjectName(u"label_title3")
        self.label_title3.setGeometry(QRect(1, 1, 478, 60))
        sizePolicy1.setHeightForWidth(self.label_title3.sizePolicy().hasHeightForWidth())
        self.label_title3.setSizePolicy(sizePolicy1)
        self.label_title3.setBaseSize(QSize(0, 0))
        self.label_title3.setFont(font3)
        self.label_title3.setStyleSheet(u"QLabel {\n"
"	background-color: deepskyblue;\n"
"	color: white;\n"
"	border: none;\n"
"	border-top-left-radius :5px;\n"
"	border-top-right-radius : 5px;\n"
"	border-bottom-left-radius : 0px;\n"
"	border-bottom-right-radius : 0px;\n"
"}")
        self.label_title3.setLineWidth(0)
        self.label_title3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_prev2 = QPushButton(self.frame3)
        self.button_prev2.setObjectName(u"button_prev2")
        self.button_prev2.setGeometry(QRect(300, 410, 80, 30))
        sizePolicy.setHeightForWidth(self.button_prev2.sizePolicy().hasHeightForWidth())
        self.button_prev2.setSizePolicy(sizePolicy)
        self.button_prev2.setFont(font2)
        self.button_prev2.setAutoFillBackground(False)
        self.button_prev2.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_prev2.setIcon(icon)
        self.button_prev2.setCheckable(False)
        self.button_prev2.setAutoExclusive(False)
        self.button_prev2.setFlat(False)
        self.label_preview3 = QLabel(self.frame3)
        self.label_preview3.setObjectName(u"label_preview3")
        self.label_preview3.setGeometry(QRect(70, 130, 340, 240))
        self.label_preview3.setMinimumSize(QSize(0, 0))
        self.label_preview3.setFont(font6)
        self.label_preview3.setStyleSheet(u"QLabel {\n"
"	color: black;\n"
"	background-color: #EFEFEF;\n"
"	border-radius: 0px;\n"
"	border: 1px solid #8f8f91;\n"
"}")
        self.label_preview3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_step3 = QLabel(self.frame3)
        self.label_step3.setObjectName(u"label_step3")
        self.label_step3.setGeometry(QRect(80, 80, 80, 30))
        self.label_step3.setFont(font5)
        self.label_step3.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :15px;\n"
"	border-top-right-radius : 0px;\n"
"	border-bottom-left-radius :15px;\n"
"	border-bottom-right-radius : 0px;\n"
"	border: none;\n"
"	background-color: deepskyblue;\n"
"	color: white;\n"
"}")
        self.label_step3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_step3_text = QLabel(self.frame3)
        self.label_step3_text.setObjectName(u"label_step3_text")
        self.label_step3_text.setGeometry(QRect(160, 80, 240, 30))
        self.label_step3_text.setFont(font4)
        self.label_step3_text.setStyleSheet(u"QLabel{\n"
"	border-top-left-radius :0px;\n"
"	border-top-right-radius : 15px;\n"
"	border-bottom-left-radius :0px;\n"
"	border-bottom-right-radius : 15px;\n"
"	border: none;\n"
"	background-color: #EFEFEF;\n"
"	color: black;\n"
"}")
        self.label_step3_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_tip2 = QLabel(self.frame3)
        self.label_tip2.setObjectName(u"label_tip2")
        self.label_tip2.setGeometry(QRect(70, 370, 340, 20))
        self.label_tip2.setFont(font7)
        self.label_tip2.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_tip2.setAlignment(Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignHCenter)
        self.stackedWidget.addWidget(self.page3)

        self.retranslateUi(ActionWizardWindow)

        self.stackedWidget.setCurrentIndex(0)
        self.button_next1.setDefault(False)
        self.button_cancel.setDefault(False)
        self.button_next2.setDefault(False)
        self.button_prev1.setDefault(False)
        self.button_save.setDefault(False)
        self.button_prev2.setDefault(False)


        QMetaObject.connectSlotsByName(ActionWizardWindow)
    # setupUi

    def retranslateUi(self, ActionWizardWindow):
        ActionWizardWindow.setWindowTitle(QCoreApplication.translate("ActionWizardWindow", u"DeepOrder", None))
        self.button_next1.setText(QCoreApplication.translate("ActionWizardWindow", u"\ub2e4\uc74c", None))
        self.label_title1.setText(QCoreApplication.translate("ActionWizardWindow", u"Action Name", None))
        self.label_preview1.setText(QCoreApplication.translate("ActionWizardWindow", u"\uc774\uacf3\uc744 \ud074\ub9ad\ud558\uc5ec\n"
"\uc774\ubbf8\uc9c0\ub97c \uc120\ud0dd\ud558\uc138\uc694", None))
        self.label_step1.setText(QCoreApplication.translate("ActionWizardWindow", u"STEP 1", None))
        self.label_step1_text.setText(QCoreApplication.translate("ActionWizardWindow", u"\uc8fc\ubb38 \ud31d\uc5c5\uc758 \uc2a4\ud06c\ub9b0\uc0f7\uc744 \ucc0d\uc5b4 \uc774\ubbf8\uc9c0\ub97c \ucca8\ubd80\ud558\uc138\uc694", None))
        self.button_cancel.setText(QCoreApplication.translate("ActionWizardWindow", u"\ucde8\uc18c", None))
        self.button_next2.setText(QCoreApplication.translate("ActionWizardWindow", u"\ub2e4\uc74c", None))
        self.label_title2.setText(QCoreApplication.translate("ActionWizardWindow", u"Action Name", None))
        self.button_prev1.setText(QCoreApplication.translate("ActionWizardWindow", u"\uc774\uc804", None))
        self.label_preview2.setText("")
        self.label_step2.setText(QCoreApplication.translate("ActionWizardWindow", u"STEP 2", None))
        self.label_step2_text.setText(QCoreApplication.translate("ActionWizardWindow", u"\uc2dc\uac04\uc870\uc815 \uc694\uc18c\uc758 \uc704\uce58\ub97c \uc9c0\uc815\ud574\uc8fc\uc138\uc694", None))
        self.label_tip1.setText(QCoreApplication.translate("ActionWizardWindow", u"\uc774\ubbf8\uc9c0\ub97c \ud074\ub9ad\ud558\uc5ec \uc704\uce58\ub97c \uc9c0\uc815\ud558\uc138\uc694", None))
        self.button_save.setText(QCoreApplication.translate("ActionWizardWindow", u"\uc644\ub8cc", None))
        self.label_title3.setText(QCoreApplication.translate("ActionWizardWindow", u"Action Name", None))
        self.button_prev2.setText(QCoreApplication.translate("ActionWizardWindow", u"\uc774\uc804", None))
        self.label_preview3.setText("")
        self.label_step3.setText(QCoreApplication.translate("ActionWizardWindow", u"STEP 3", None))
        self.label_step3_text.setText(QCoreApplication.translate("ActionWizardWindow", u"\uc218\ub77d/\uac70\uc808 \ubc84\ud2bc\uc758 \uc704\uce58\ub97c \uc9c0\uc815\ud574\uc8fc\uc138\uc694", None))
        self.label_tip2.setText(QCoreApplication.translate("ActionWizardWindow", u"\uc774\ubbf8\uc9c0\ub97c \ud074\ub9ad\ud558\uc5ec \uc704\uce58\ub97c \uc9c0\uc815\ud558\uc138\uc694", None))
    # retranslateUi

