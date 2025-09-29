# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ActionWindow.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog,
    QFrame, QHeaderView, QLabel, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QWidget)

class Ui_ActionWindow(object):
    def setupUi(self, ActionWindow):
        if not ActionWindow.objectName():
            ActionWindow.setObjectName(u"ActionWindow")
        ActionWindow.resize(500, 570)
        ActionWindow.setMaximumSize(QSize(500, 700))
        font = QFont()
        font.setFamilies([u"Arial"])
        ActionWindow.setFont(font)
        ActionWindow.setAutoFillBackground(False)
        ActionWindow.setStyleSheet(u"QDialog{\n"
"	background-color: #EFEFEF;\n"
"}")
        self.actionwindow = QWidget(ActionWindow)
        self.actionwindow.setObjectName(u"actionwindow")
        font1 = QFont()
        font1.setBold(False)
        self.actionwindow.setFont(font1)
        self.frame_main = QFrame(self.actionwindow)
        self.frame_main.setObjectName(u"frame_main")
        self.frame_main.setGeometry(QRect(10, 10, 480, 550))
        self.frame_main.setFont(font)
        self.frame_main.setStyleSheet(u"background: white;\n"
"border:1px solid black;\n"
"border-radius: 5px;\n"
"")
        self.frame_main.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_main.setFrameShadow(QFrame.Shadow.Raised)
        self.tableWidget = QTableWidget(self.frame_main)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setGeometry(QRect(20, 80, 340, 340))
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(12)
        self.tableWidget.setFont(font2)
        self.tableWidget.setStyleSheet(u"QTableWidget {\n"
"    color: black;\n"
"    border: 1px solid #8f8f91;\n"
"    border-radius: 5px;\n"
"    background-color: white;\n"
"  }\n"
"  QTableWidget::item {\n"
"    border: none;\n"
"    padding: 4px;\n"
"  }\n"
"  QTableWidget::item:selected {\n"
"    background-color: #0052cc;\n"
"    color: white;\n"
"  }\n"
"  QScrollBar {\n"
"    width: 0px;\n"
"    height: 0px;\n"
"  }")
        self.tableWidget.setFrameShape(QFrame.Shape.StyledPanel)
        self.tableWidget.setFrameShadow(QFrame.Shadow.Plain)
        self.tableWidget.setLineWidth(0)
        self.tableWidget.setDragEnabled(True)
        self.tableWidget.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.tableWidget.setDefaultDropAction(Qt.DropAction.IgnoreAction)
        self.tableWidget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget.setShowGrid(False)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(50)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)
        self.button_add = QPushButton(self.frame_main)
        self.button_add.setObjectName(u"button_add")
        self.button_add.setGeometry(QRect(375, 80, 90, 40))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_add.sizePolicy().hasHeightForWidth())
        self.button_add.setSizePolicy(sizePolicy)
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(10)
        font3.setBold(False)
        font3.setHintingPreference(QFont.PreferVerticalHinting)
        self.button_add.setFont(font3)
        self.button_add.setAutoFillBackground(False)
        self.button_add.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.InsertText))
        self.button_add.setIcon(icon)
        self.button_add.setAutoExclusive(False)
        self.button_add.setFlat(False)
        self.button_delete = QPushButton(self.frame_main)
        self.button_delete.setObjectName(u"button_delete")
        self.button_delete.setGeometry(QRect(375, 320, 90, 40))
        sizePolicy.setHeightForWidth(self.button_delete.sizePolicy().hasHeightForWidth())
        self.button_delete.setSizePolicy(sizePolicy)
        self.button_delete.setFont(font3)
        self.button_delete.setAutoFillBackground(False)
        self.button_delete.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_delete.setIcon(icon)
        self.button_delete.setAutoExclusive(False)
        self.button_delete.setFlat(False)
        self.button_edit = QPushButton(self.frame_main)
        self.button_edit.setObjectName(u"button_edit")
        self.button_edit.setGeometry(QRect(375, 260, 90, 40))
        sizePolicy.setHeightForWidth(self.button_edit.sizePolicy().hasHeightForWidth())
        self.button_edit.setSizePolicy(sizePolicy)
        self.button_edit.setFont(font3)
        self.button_edit.setAutoFillBackground(False)
        self.button_edit.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_edit.setIcon(icon)
        self.button_edit.setAutoExclusive(False)
        self.button_edit.setFlat(False)
        self.button_delay = QPushButton(self.frame_main)
        self.button_delay.setObjectName(u"button_delay")
        self.button_delay.setGeometry(QRect(375, 140, 90, 40))
        sizePolicy.setHeightForWidth(self.button_delay.sizePolicy().hasHeightForWidth())
        self.button_delay.setSizePolicy(sizePolicy)
        self.button_delay.setFont(font3)
        self.button_delay.setAutoFillBackground(False)
        self.button_delay.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_delay.setIcon(icon)
        self.button_delay.setAutoExclusive(False)
        self.button_delay.setFlat(False)
        self.button_save = QPushButton(self.frame_main)
        self.button_save.setObjectName(u"button_save")
        self.button_save.setGeometry(QRect(300, 510, 80, 30))
        sizePolicy.setHeightForWidth(self.button_save.sizePolicy().hasHeightForWidth())
        self.button_save.setSizePolicy(sizePolicy)
        self.button_save.setFont(font3)
        self.button_save.setAutoFillBackground(False)
        self.button_save.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_save.setIcon(icon)
        self.button_save.setAutoExclusive(False)
        self.button_save.setFlat(False)
        self.button_cancel = QPushButton(self.frame_main)
        self.button_cancel.setObjectName(u"button_cancel")
        self.button_cancel.setGeometry(QRect(390, 510, 80, 30))
        sizePolicy.setHeightForWidth(self.button_cancel.sizePolicy().hasHeightForWidth())
        self.button_cancel.setSizePolicy(sizePolicy)
        self.button_cancel.setFont(font3)
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
        self.button_down = QPushButton(self.frame_main)
        self.button_down.setObjectName(u"button_down")
        self.button_down.setGeometry(QRect(335, 435, 25, 25))
        sizePolicy.setHeightForWidth(self.button_down.sizePolicy().hasHeightForWidth())
        self.button_down.setSizePolicy(sizePolicy)
        self.button_down.setFont(font3)
        self.button_down.setAutoFillBackground(False)
        self.button_down.setStyleSheet(u"QPushButton {\n"
"	border-radius: 5px;\n"
"	border: 1px solid #8f8f91;\n"
"	background-color: #EFEFEF;\n"
"	color:black;\n"
"	padding: 2px 4px;\n"
"}")
        self.button_down.setIcon(icon)
        self.button_down.setAutoExclusive(False)
        self.button_down.setFlat(False)
        self.button_up = QPushButton(self.frame_main)
        self.button_up.setObjectName(u"button_up")
        self.button_up.setGeometry(QRect(305, 435, 25, 25))
        sizePolicy.setHeightForWidth(self.button_up.sizePolicy().hasHeightForWidth())
        self.button_up.setSizePolicy(sizePolicy)
        self.button_up.setFont(font3)
        self.button_up.setAutoFillBackground(False)
        self.button_up.setStyleSheet(u"QPushButton {\n"
"	border-radius: 5px;\n"
"	border: 1px solid #8f8f91;\n"
"	background-color: #EFEFEF;\n"
"	color:black;\n"
"	padding: 2px 4px;\n"
"}")
        self.button_up.setIcon(icon)
        self.button_up.setAutoExclusive(False)
        self.button_up.setFlat(False)
        self.label_title = QLabel(self.frame_main)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setGeometry(QRect(1, 1, 478, 60))
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(128)
        sizePolicy1.setVerticalStretch(60)
        sizePolicy1.setHeightForWidth(self.label_title.sizePolicy().hasHeightForWidth())
        self.label_title.setSizePolicy(sizePolicy1)
        self.label_title.setBaseSize(QSize(0, 0))
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(20)
        font4.setBold(True)
        self.label_title.setFont(font4)
        self.label_title.setStyleSheet(u"background-color: deepskyblue;\n"
"color: white;\n"
"border: none;\n"
"border-top-left-radius :5px;\n"
"border-top-right-radius : 5px;\n"
"border-bottom-left-radius : 0px;\n"
"border-bottom-right-radius : 0px;")
        self.label_title.setLineWidth(0)
        self.label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_priority = QPushButton(self.frame_main)
        self.button_priority.setObjectName(u"button_priority")
        self.button_priority.setGeometry(QRect(375, 200, 90, 40))
        sizePolicy.setHeightForWidth(self.button_priority.sizePolicy().hasHeightForWidth())
        self.button_priority.setSizePolicy(sizePolicy)
        self.button_priority.setFont(font3)
        self.button_priority.setAutoFillBackground(False)
        self.button_priority.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_priority.setIcon(icon)
        self.button_priority.setAutoExclusive(False)
        self.button_priority.setFlat(False)
        self.button_setting = QPushButton(self.frame_main)
        self.button_setting.setObjectName(u"button_setting")
        self.button_setting.setGeometry(QRect(375, 380, 90, 40))
        sizePolicy.setHeightForWidth(self.button_setting.sizePolicy().hasHeightForWidth())
        self.button_setting.setSizePolicy(sizePolicy)
        self.button_setting.setFont(font3)
        self.button_setting.setAutoFillBackground(False)
        self.button_setting.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_setting.setIcon(icon)
        self.button_setting.setAutoExclusive(False)
        self.button_setting.setFlat(False)
        self.comboBox_window = QComboBox(self.frame_main)
        self.comboBox_window.setObjectName(u"comboBox_window")
        self.comboBox_window.setGeometry(QRect(90, 435, 200, 25))
        font5 = QFont()
        font5.setFamilies([u"Arial"])
        font5.setPointSize(10)
        self.comboBox_window.setFont(font5)
        self.comboBox_window.setStyleSheet(u"QComboBox {\n"
"    border: 1px solid #8f8f91;\n"
"    border-radius: 3px;\n"
"    padding: 1px 3px 1px 3px;\n"
"    background: white;\n"
"    selection-color: white;\n"
"    selection-background-color: blue;\n"
"}")
        self.button_arrowup = QPushButton(self.frame_main)
        self.button_arrowup.setObjectName(u"button_arrowup")
        self.button_arrowup.setGeometry(QRect(182, 95, 16, 16))
        self.button_arrowup.setMinimumSize(QSize(0, 0))
        self.button_arrowup.setMaximumSize(QSize(88, 16777215))
        self.button_arrowup.setFont(font2)
        self.button_arrowup.setStyleSheet(u"color: #8f8f91;\n"
"background: transparent;\n"
"padding: 2px;\n"
"border:none;")
        self.button_arrowdown = QPushButton(self.frame_main)
        self.button_arrowdown.setObjectName(u"button_arrowdown")
        self.button_arrowdown.setGeometry(QRect(182, 390, 16, 16))
        self.button_arrowdown.setMinimumSize(QSize(0, 0))
        self.button_arrowdown.setMaximumSize(QSize(88, 16777215))
        self.button_arrowdown.setFont(font2)
        self.button_arrowdown.setStyleSheet(u"color: #8f8f91;\n"
"background: transparent;\n"
"padding: 2px;\n"
"border:none;")
        self.label_pgselect = QLabel(self.frame_main)
        self.label_pgselect.setObjectName(u"label_pgselect")
        self.label_pgselect.setGeometry(QRect(23, 435, 60, 25))
        self.label_pgselect.setFont(font5)
        self.label_pgselect.setStyleSheet(u"border:none;\n"
"color:black;")
        self.label_pgselect.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.retranslateUi(ActionWindow)

        self.button_add.setDefault(False)
        self.button_delete.setDefault(False)
        self.button_edit.setDefault(False)
        self.button_delay.setDefault(False)
        self.button_save.setDefault(False)
        self.button_cancel.setDefault(False)
        self.button_down.setDefault(False)
        self.button_up.setDefault(False)
        self.button_priority.setDefault(False)
        self.button_setting.setDefault(False)


        QMetaObject.connectSlotsByName(ActionWindow)
    # setupUi

    def retranslateUi(self, ActionWindow):
        ActionWindow.setWindowTitle(QCoreApplication.translate("ActionWindow", u"\ub3d9\uc791 \ud3b8\uc9d1", None))
        self.button_add.setText(QCoreApplication.translate("ActionWindow", u"\ub3d9\uc791 \ucd94\uac00", None))
        self.button_delete.setText(QCoreApplication.translate("ActionWindow", u"\uc0ad\uc81c", None))
        self.button_edit.setText(QCoreApplication.translate("ActionWindow", u"\uc218\uc815", None))
        self.button_delay.setText(QCoreApplication.translate("ActionWindow", u"\ub300\uae30\uc2dc\uac04 \ucd94\uac00", None))
        self.button_save.setText(QCoreApplication.translate("ActionWindow", u"\uc800\uc7a5", None))
        self.button_cancel.setText(QCoreApplication.translate("ActionWindow", u"\ucde8\uc18c", None))
        self.button_down.setText(QCoreApplication.translate("ActionWindow", u"\u2228", None))
        self.button_up.setText(QCoreApplication.translate("ActionWindow", u"\u2227", None))
        self.label_title.setText(QCoreApplication.translate("ActionWindow", u"macro name", None))
        self.button_priority.setText(QCoreApplication.translate("ActionWindow", u"\uc6b0\uc120\uc21c\uc704 \uc9c0\uc815", None))
        self.button_setting.setText(QCoreApplication.translate("ActionWindow", u"\uc124\uc815", None))
        self.comboBox_window.setPlaceholderText(QCoreApplication.translate("ActionWindow", u"\uc120\ud0dd\ud558\uc138\uc694", None))
        self.button_arrowup.setText(QCoreApplication.translate("ActionWindow", u"\u25b2", None))
        self.button_arrowdown.setText(QCoreApplication.translate("ActionWindow", u"\u25bc", None))
        self.label_pgselect.setText(QCoreApplication.translate("ActionWindow", u"\ud504\ub85c\uadf8\ub7a8 \uc120\ud0dd :", None))
    # retranslateUi

