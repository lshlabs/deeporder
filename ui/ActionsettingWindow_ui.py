# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ActionsettingWindow.ui'
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
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox,
    QDialog, QFrame, QLabel, QPushButton,
    QSizePolicy, QSpinBox, QTabWidget, QWidget)

class Ui_ActionSettingWindow(object):
    def setupUi(self, ActionSettingWindow):
        if not ActionSettingWindow.objectName():
            ActionSettingWindow.setObjectName(u"ActionSettingWindow")
        ActionSettingWindow.resize(360, 400)
        font = QFont()
        font.setFamilies([u"Arial"])
        ActionSettingWindow.setFont(font)
        ActionSettingWindow.setStyleSheet(u"QMainWindow{\n"
"	background-color: #EFEFEF;\n"
"}")
        ActionSettingWindow.setSizeGripEnabled(False)
        ActionSettingWindow.setModal(True)
        self.frame_main = QFrame(ActionSettingWindow)
        self.frame_main.setObjectName(u"frame_main")
        self.frame_main.setGeometry(QRect(10, 10, 340, 380))
        self.frame_main.setFont(font)
        self.frame_main.setStyleSheet(u"background: white;\n"
"border:1px solid black;\n"
"border-radius: 5px;\n"
"")
        self.frame_main.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_main.setFrameShadow(QFrame.Shadow.Raised)
        self.button_save = QPushButton(self.frame_main)
        self.button_save.setObjectName(u"button_save")
        self.button_save.setGeometry(QRect(160, 340, 80, 30))
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
        self.button_cancel.setGeometry(QRect(250, 340, 80, 30))
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
        self.tabWidget = QTabWidget(self.frame_main)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(1, 1, 338, 320))
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(12)
        self.tabWidget.setFont(font2)
        self.tabWidget.setStyleSheet(u"border:none;\n"
"\n"
"QTabBar::tab {\n"
"  background: rgb(230, 230, 230); \n"
"  border: 1px solid lightgray; \n"
"  padding: 15px;\n"
"} ")
        self.tabWidget.setTabShape(QTabWidget.TabShape.Rounded)
        self.tabWidget.setElideMode(Qt.TextElideMode.ElideLeft)
        self.tabWidget.setUsesScrollButtons(False)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tab1 = QWidget()
        self.tab1.setObjectName(u"tab1")
        self.label_stop = QLabel(self.tab1)
        self.label_stop.setObjectName(u"label_stop")
        self.label_stop.setGeometry(QRect(20, 140, 100, 20))
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(10)
        self.label_stop.setFont(font3)
        self.label_stop.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_stop.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.comboBox_run = QComboBox(self.tab1)
        self.comboBox_run.setObjectName(u"comboBox_run")
        self.comboBox_run.setGeometry(QRect(20, 80, 110, 25))
        self.comboBox_run.setFont(font3)
        self.comboBox_run.setStyleSheet(u"QComboBox\n"
"{\n"
"  selection-color: white;\n"
"  selection-background-color: blue;\n"
"  color:black;\n"
"  border:1px solid black;\n"
"  border-radius: 5px;\n"
"  padding-left: 3px;\n"
"}")
        self.label_run = QLabel(self.tab1)
        self.label_run.setObjectName(u"label_run")
        self.label_run.setGeometry(QRect(20, 50, 100, 20))
        self.label_run.setFont(font3)
        self.label_run.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_run.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.comboBox_stop = QComboBox(self.tab1)
        self.comboBox_stop.setObjectName(u"comboBox_stop")
        self.comboBox_stop.setGeometry(QRect(20, 170, 110, 25))
        self.comboBox_stop.setFont(font3)
        self.comboBox_stop.setStyleSheet(u"QComboBox\n"
"{\n"
"  selection-color: white;\n"
"  selection-background-color: blue;\n"
"  color:black;\n"
"  border:1px solid black;\n"
"  border-radius: 5px;\n"
"  padding-left: 3px;\n"
"}")
        self.checkBox_alt1 = QCheckBox(self.tab1)
        self.checkBox_alt1.setObjectName(u"checkBox_alt1")
        self.checkBox_alt1.setGeometry(QRect(200, 80, 50, 25))
        self.checkBox_alt1.setStyleSheet(u"border: none;")
        self.checkBox_shift2 = QCheckBox(self.tab1)
        self.checkBox_shift2.setObjectName(u"checkBox_shift2")
        self.checkBox_shift2.setGeometry(QRect(250, 170, 50, 25))
        self.checkBox_shift2.setStyleSheet(u"border: none;")
        self.checkBox_ctrl2 = QCheckBox(self.tab1)
        self.checkBox_ctrl2.setObjectName(u"checkBox_ctrl2")
        self.checkBox_ctrl2.setGeometry(QRect(150, 170, 50, 25))
        self.checkBox_ctrl2.setStyleSheet(u"border: none;")
        self.checkBox_alt2 = QCheckBox(self.tab1)
        self.checkBox_alt2.setObjectName(u"checkBox_alt2")
        self.checkBox_alt2.setGeometry(QRect(200, 170, 50, 25))
        self.checkBox_alt2.setStyleSheet(u"border: none;")
        self.label_tip = QLabel(self.tab1)
        self.label_tip.setObjectName(u"label_tip")
        self.label_tip.setGeometry(QRect(10, -20, 300, 20))
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(9)
        self.label_tip.setFont(font4)
        self.label_tip.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.checkBox_shift1 = QCheckBox(self.tab1)
        self.checkBox_shift1.setObjectName(u"checkBox_shift1")
        self.checkBox_shift1.setGeometry(QRect(250, 80, 50, 25))
        self.checkBox_shift1.setStyleSheet(u"border: none;")
        self.checkBox_ctrl1 = QCheckBox(self.tab1)
        self.checkBox_ctrl1.setObjectName(u"checkBox_ctrl1")
        self.checkBox_ctrl1.setGeometry(QRect(150, 80, 50, 25))
        self.checkBox_ctrl1.setStyleSheet(u"border: none;")
        self.tabWidget.addTab(self.tab1, "")
        self.tab2 = QWidget()
        self.tab2.setObjectName(u"tab2")
        self.label3 = QLabel(self.tab2)
        self.label3.setObjectName(u"label3")
        self.label3.setGeometry(QRect(29, 130, 70, 30))
        self.label3.setFont(font3)
        self.label3.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label4 = QLabel(self.tab2)
        self.label4.setObjectName(u"label4")
        self.label4.setGeometry(QRect(220, 130, 60, 30))
        self.label4.setFont(font3)
        self.label4.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label4.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.spinBox_stack = QSpinBox(self.tab2)
        self.spinBox_stack.setObjectName(u"spinBox_stack")
        self.spinBox_stack.setGeometry(QRect(110, 130, 100, 30))
        self.spinBox_stack.setFont(font)
        self.spinBox_stack.setStyleSheet(u"QSpinBox{\n"
"	border: 1px solid black;\n"
"	padding-left:3px;\n"
"}\n"
"\n"
"\n"
"\n"
"QSpinBox::down-button{\n"
"	width: 20px;\n"
"	margin-right:1px;\n"
"}\n"
"\n"
"QSpinBox::up-button{\n"
"	width: 20px;\n"
"	margin-right:1px;\n"
"}")
        self.spinBox_stack.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.spinBox_stack.setMaximum(100)
        self.spinBox_stack.setValue(5)
        self.spinBox_default = QSpinBox(self.tab2)
        self.spinBox_default.setObjectName(u"spinBox_default")
        self.spinBox_default.setGeometry(QRect(110, 60, 100, 30))
        self.spinBox_default.setFont(font)
        self.spinBox_default.setStyleSheet(u"QSpinBox{\n"
"	border: 1px solid black;\n"
"	padding-left:3px;\n"
"}\n"
"\n"
"\n"
"\n"
"QSpinBox::down-button{\n"
"	width: 20px;\n"
"	margin-right:1px;\n"
"}\n"
"\n"
"QSpinBox::up-button{\n"
"	width: 20px;\n"
"	margin-right:1px;\n"
"}")
        self.spinBox_default.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.spinBox_default.setMaximum(100)
        self.spinBox_default.setValue(15)
        self.label1 = QLabel(self.tab2)
        self.label1.setObjectName(u"label1")
        self.label1.setGeometry(QRect(30, 60, 70, 30))
        self.label1.setFont(font3)
        self.label1.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label2 = QLabel(self.tab2)
        self.label2.setObjectName(u"label2")
        self.label2.setGeometry(QRect(220, 60, 61, 30))
        self.label2.setFont(font3)
        self.label2.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label2.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.label6 = QLabel(self.tab2)
        self.label6.setObjectName(u"label6")
        self.label6.setGeometry(QRect(220, 175, 60, 30))
        self.label6.setFont(font3)
        self.label6.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label6.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.spinBox_delay = QSpinBox(self.tab2)
        self.spinBox_delay.setObjectName(u"spinBox_delay")
        self.spinBox_delay.setGeometry(QRect(110, 175, 100, 30))
        self.spinBox_delay.setFont(font)
        self.spinBox_delay.setStyleSheet(u"QSpinBox{\n"
"	border: 1px solid black;\n"
"	padding-left:3px;\n"
"}\n"
"\n"
"\n"
"\n"
"QSpinBox::down-button{\n"
"	width: 20px;\n"
"	margin-right:1px;\n"
"}\n"
"\n"
"QSpinBox::up-button{\n"
"	width: 20px;\n"
"	margin-right:1px;\n"
"}")
        self.spinBox_delay.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.spinBox_delay.setMaximum(100)
        self.spinBox_delay.setValue(20)
        self.label5 = QLabel(self.tab2)
        self.label5.setObjectName(u"label5")
        self.label5.setGeometry(QRect(39, 175, 61, 30))
        self.label5.setFont(font3)
        self.label5.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label5.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.tabWidget.addTab(self.tab2, "")

        self.retranslateUi(ActionSettingWindow)

        self.button_save.setDefault(False)
        self.button_cancel.setDefault(False)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(ActionSettingWindow)
    # setupUi

    def retranslateUi(self, ActionSettingWindow):
        ActionSettingWindow.setWindowTitle(QCoreApplication.translate("ActionSettingWindow", u"\ub3d9\uc791 \uc124\uc815", None))
        self.button_save.setText(QCoreApplication.translate("ActionSettingWindow", u"\uc800\uc7a5", None))
        self.button_cancel.setText(QCoreApplication.translate("ActionSettingWindow", u"\ucde8\uc18c", None))
        self.label_stop.setText(QCoreApplication.translate("ActionSettingWindow", u"\u25cf \uc911\uc9c0 \ub2e8\ucd95\ud0a4", None))
        self.comboBox_run.setPlaceholderText(QCoreApplication.translate("ActionSettingWindow", u"\uc120\ud0dd\ud558\uc138\uc694", None))
        self.label_run.setText(QCoreApplication.translate("ActionSettingWindow", u"\u25cf \uc2dc\uc791 \ub2e8\ucd95\ud0a4", None))
        self.comboBox_stop.setPlaceholderText(QCoreApplication.translate("ActionSettingWindow", u"\uc120\ud0dd\ud558\uc138\uc694", None))
        self.checkBox_alt1.setText(QCoreApplication.translate("ActionSettingWindow", u"Alt", None))
        self.checkBox_shift2.setText(QCoreApplication.translate("ActionSettingWindow", u"Shift", None))
        self.checkBox_ctrl2.setText(QCoreApplication.translate("ActionSettingWindow", u"Ctrl", None))
        self.checkBox_alt2.setText(QCoreApplication.translate("ActionSettingWindow", u"Alt", None))
        self.label_tip.setText(QCoreApplication.translate("ActionSettingWindow", u"\ub3d9\uc791\uc758 \uc2dc\uc791/\uc911\uc9c0 \ub2e8\ucd95\ud0a4\ub97c \uc124\uc815\ud558\uc138\uc694", None))
        self.checkBox_shift1.setText(QCoreApplication.translate("ActionSettingWindow", u"Shift", None))
        self.checkBox_ctrl1.setText(QCoreApplication.translate("ActionSettingWindow", u"Ctrl", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), QCoreApplication.translate("ActionSettingWindow", u"\ub2e8\ucd95\ud0a4 \uc124\uc815", None))
        self.label3.setText(QCoreApplication.translate("ActionSettingWindow", u"\u25cf \uc9c4\ud589\uc911\uc778 \uc8fc\ubb38", None))
        self.label4.setText(QCoreApplication.translate("ActionSettingWindow", u"\uac1c \uc774\uc0c1\uc77c \uc2dc", None))
        self.spinBox_stack.setSpecialValueText("")
        self.spinBox_default.setSpecialValueText("")
        self.label1.setText(QCoreApplication.translate("ActionSettingWindow", u"\u25cf \uae30\ubcf8 \uc124\uc815 \uc2dc\uac04", None))
        self.label2.setText(QCoreApplication.translate("ActionSettingWindow", u"\ubd84", None))
        self.label6.setText(QCoreApplication.translate("ActionSettingWindow", u"\ubd84", None))
        self.spinBox_delay.setSpecialValueText("")
        self.spinBox_delay.setSuffix("")
        self.spinBox_delay.setPrefix("")
        self.label5.setText(QCoreApplication.translate("ActionSettingWindow", u"\uc124\uc815\uc2dc\uac04", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab2), QCoreApplication.translate("ActionSettingWindow", u"\ubc30\ub2ec\uc2dc\uac04 \uc124\uc815", None))
    # retranslateUi

