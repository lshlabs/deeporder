# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QLabel,
    QLineEdit, QListView, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QSizePolicy, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(500, 570)
        MainWindow.setMaximumSize(QSize(500, 700))
        font = QFont()
        font.setFamilies([u"Arial"])
        MainWindow.setFont(font)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet(u"QMainWindow{\n"
"	background-color: #EFEFEF;\n"
"}")
        self.mainwindow = QWidget(MainWindow)
        self.mainwindow.setObjectName(u"mainwindow")
        font1 = QFont()
        font1.setBold(False)
        self.mainwindow.setFont(font1)
        self.frame_main = QFrame(self.mainwindow)
        self.frame_main.setObjectName(u"frame_main")
        self.frame_main.setGeometry(QRect(10, 10, 480, 550))
        self.frame_main.setFont(font)
        self.frame_main.setStyleSheet(u"background: white;\n"
"border:1px solid black;\n"
"border-radius: 5px;\n"
"")
        self.frame_main.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_main.setFrameShadow(QFrame.Shadow.Raised)
        self.label_title = QLabel(self.frame_main)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setGeometry(QRect(1, 1, 478, 60))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(128)
        sizePolicy.setVerticalStretch(60)
        sizePolicy.setHeightForWidth(self.label_title.sizePolicy().hasHeightForWidth())
        self.label_title.setSizePolicy(sizePolicy)
        self.label_title.setBaseSize(QSize(0, 0))
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(20)
        font2.setBold(True)
        self.label_title.setFont(font2)
        self.label_title.setStyleSheet(u"QLabel {\n"
"	background-color: darkgray;\n"
"	color: white;\n"
"	border: none;\n"
"	border-top-left-radius :5px;\n"
"	border-top-right-radius : 5px;\n"
"	border-bottom-left-radius : 0px;\n"
"	border-bottom-right-radius : 0px;\n"
"}")
        self.label_title.setLineWidth(0)
        self.label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.listWidget = QListWidget(self.frame_main)
        self.listWidget.setObjectName(u"listWidget")
        self.listWidget.setGeometry(QRect(30, 80, 420, 300))
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(12)
        font3.setKerning(True)
        self.listWidget.setFont(font3)
        self.listWidget.setStyleSheet(u"QListWidget {\n"
"                color: black;\n"
"                border-radius: 5px;\n"
"            }\n"
"QListWidget::item {\n"
"                height: 30px;\n"
"                padding: 4px;\n"
"}")
        self.listWidget.setFrameShadow(QFrame.Shadow.Plain)
        self.listWidget.setAutoScroll(True)
        self.listWidget.setAutoScrollMargin(16)
        self.listWidget.setDragEnabled(True)
        self.listWidget.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.listWidget.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.listWidget.setIconSize(QSize(0, 0))
        self.listWidget.setMovement(QListView.Movement.Static)
        self.listWidget.setLayoutMode(QListView.LayoutMode.SinglePass)
        self.listWidget.setSpacing(3)
        self.listWidget.setViewMode(QListView.ViewMode.ListMode)
        self.listWidget.setItemAlignment(Qt.AlignmentFlag.AlignBaseline)
        self.listWidget.setSortingEnabled(False)
        self.lineEdit = QLineEdit(self.frame_main)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(30, 400, 300, 30))
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(10)
        self.lineEdit.setFont(font4)
        self.lineEdit.setStyleSheet(u"QLineEdit {\n"
"	color:black;\n"
"	border-radius: 5px; \n"
"	padding-left:10px;\n"
"}")
        self.lineEdit.setMaxLength(32767)
        self.button_add = QPushButton(self.frame_main)
        self.button_add.setObjectName(u"button_add")
        self.button_add.setGeometry(QRect(340, 400, 50, 30))
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.button_add.sizePolicy().hasHeightForWidth())
        self.button_add.setSizePolicy(sizePolicy1)
        font5 = QFont()
        font5.setFamilies([u"Arial"])
        font5.setPointSize(10)
        font5.setBold(False)
        font5.setHintingPreference(QFont.PreferVerticalHinting)
        self.button_add.setFont(font5)
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
        self.button_delete.setGeometry(QRect(400, 400, 50, 30))
        sizePolicy1.setHeightForWidth(self.button_delete.sizePolicy().hasHeightForWidth())
        self.button_delete.setSizePolicy(sizePolicy1)
        self.button_delete.setFont(font5)
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
        self.button_edit.setGeometry(QRect(340, 440, 50, 30))
        sizePolicy1.setHeightForWidth(self.button_edit.sizePolicy().hasHeightForWidth())
        self.button_edit.setSizePolicy(sizePolicy1)
        self.button_edit.setFont(font5)
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
        self.label_run = QLabel(self.frame_main)
        self.label_run.setObjectName(u"label_run")
        self.label_run.setGeometry(QRect(30, 440, 150, 70))
        sizePolicy.setHeightForWidth(self.label_run.sizePolicy().hasHeightForWidth())
        self.label_run.setSizePolicy(sizePolicy)
        self.label_run.setBaseSize(QSize(0, 0))
        self.label_run.setFont(font2)
        self.label_run.setStyleSheet(u"QLabel {\n"
"	background-color: darkgray;\n"
"	color: white;\n"
"	border-top: 1px solid black;\n"
"	border-bottom: 1px solid black;\n"
"	border-left: 1px solid black;\n"
"	border-right: none;\n"
"	border-top-left-radius :5px;\n"
"	border-top-right-radius : 0px;\n"
"	border-bottom-left-radius : 5px;\n"
"	border-bottom-right-radius : 0px;\n"
"}")
        self.label_run.setLineWidth(0)
        self.label_run.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_stop = QLabel(self.frame_main)
        self.label_stop.setObjectName(u"label_stop")
        self.label_stop.setGeometry(QRect(180, 440, 150, 70))
        sizePolicy.setHeightForWidth(self.label_stop.sizePolicy().hasHeightForWidth())
        self.label_stop.setSizePolicy(sizePolicy)
        self.label_stop.setBaseSize(QSize(0, 0))
        self.label_stop.setFont(font2)
        self.label_stop.setStyleSheet(u"QLabel {\n"
"	background-color: deepskyblue;\n"
"	color: white;\n"
"	border-top: 1px solid black;\n"
"	border-bottom: 1px solid black;\n"
"	border-left: none;\n"
"	border-right: 1px solid black;\n"
"	border-top-left-radius :0px;\n"
"	border-top-right-radius : 5px;\n"
"	border-bottom-left-radius : 0px;\n"
"	border-bottom-right-radius : 5px;\n"
"}")
        self.label_stop.setLineWidth(0)
        self.label_stop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_version = QLabel(self.frame_main)
        self.label_version.setObjectName(u"label_version")
        self.label_version.setGeometry(QRect(10, 530, 130, 10))
        font6 = QFont()
        font6.setFamilies([u"Arial"])
        font6.setPointSize(9)
        self.label_version.setFont(font6)
        self.label_version.setStyleSheet(u"QLabel {\n"
"	border: none;\n"
"	color: black;\n"
"}")
        self.label_version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_copy = QPushButton(self.frame_main)
        self.button_copy.setObjectName(u"button_copy")
        self.button_copy.setGeometry(QRect(400, 440, 50, 30))
        sizePolicy1.setHeightForWidth(self.button_copy.sizePolicy().hasHeightForWidth())
        self.button_copy.setSizePolicy(sizePolicy1)
        self.button_copy.setFont(font5)
        self.button_copy.setAutoFillBackground(False)
        self.button_copy.setStyleSheet(u"QPushButton {\n"
"    border-radius: 5px;\n"
"	color:black;\n"
"    border: 1px solid #8f8f91;\n"
"    background-color: #EFEFEF;\n"
"    padding: 2px 4px;\n"
"}")
        self.button_copy.setIcon(icon)
        self.button_copy.setAutoExclusive(False)
        self.button_copy.setFlat(False)
        self.button_setting = QPushButton(self.frame_main)
        self.button_setting.setObjectName(u"button_setting")
        self.button_setting.setGeometry(QRect(340, 480, 110, 30))
        sizePolicy1.setHeightForWidth(self.button_setting.sizePolicy().hasHeightForWidth())
        self.button_setting.setSizePolicy(sizePolicy1)
        self.button_setting.setFont(font5)
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
        MainWindow.setCentralWidget(self.mainwindow)

        self.retranslateUi(MainWindow)

        self.button_add.setDefault(False)
        self.button_delete.setDefault(False)
        self.button_edit.setDefault(False)
        self.button_copy.setDefault(False)
        self.button_setting.setDefault(False)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"DeepOrder", None))
        self.label_title.setText(QCoreApplication.translate("MainWindow", u"DeepOrder", None))
        self.lineEdit.setInputMask("")
        self.lineEdit.setText("")
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\uc774\ub984\uc744 \uc785\ub825\ud558\uc138\uc694", None))
        self.button_add.setText(QCoreApplication.translate("MainWindow", u"\ucd94\uac00", None))
        self.button_delete.setText(QCoreApplication.translate("MainWindow", u"\uc0ad\uc81c", None))
        self.button_edit.setText(QCoreApplication.translate("MainWindow", u"\uc218\uc815", None))
        self.label_run.setText(QCoreApplication.translate("MainWindow", u"RUN", None))
        self.label_stop.setText(QCoreApplication.translate("MainWindow", u"STOP", None))
        self.label_version.setText(QCoreApplication.translate("MainWindow", u"v0.1.3-dev 250227", None))
        self.button_copy.setText(QCoreApplication.translate("MainWindow", u"\ubcf5\uc81c", None))
        self.button_setting.setText(QCoreApplication.translate("MainWindow", u"\uc124\uc815", None))
    # retranslateUi

