#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'first.ui'
#
# Created: Wed Mar 25 04:35:33 2015
#      by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QFileDialog
import som_v4
from som_v4 import Map
import sys
import pp_v3

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_gui(QtGui.QTabWidget):
    def __init__(self):
        QtGui.QTabWidget.__init__(self)
        self.setupUi(self)

    def setupUi(self, gui):
        gui.setObjectName(_fromUtf8("gui"))
        gui.resize(675, 371)

        self.train_tab = QtGui.QWidget()
        self.train_tab.setObjectName(_fromUtf8("train_tab"))

        self.projname_input = QtGui.QLineEdit(self.train_tab)
        self.projname_input.setGeometry(QtCore.QRect(20, 10, 291, 23))
        self.projname_input.setText(_fromUtf8(""))
        self.projname_input.setObjectName(_fromUtf8("projname_input"))

        self.dir_input = QtGui.QLineEdit(self.train_tab)
        self.dir_input.setGeometry(QtCore.QRect(20, 40, 291, 23))
        self.dir_input.setText(_fromUtf8(""))
        self.dir_input.setCursorMoveStyle(QtCore.Qt.LogicalMoveStyle)
        self.dir_input.setObjectName(_fromUtf8("dir_input"))

        self.vecdb_input = QtGui.QLineEdit(self.train_tab)
        self.vecdb_input.setGeometry(QtCore.QRect(20, 70, 291, 23))
        self.vecdb_input.setToolTip(_fromUtf8(""))
        self.vecdb_input.setObjectName(_fromUtf8("vecdb_input"))

        self.browse = QtGui.QToolButton(self.train_tab)
        self.browse.setGeometry(QtCore.QRect(320, 70, 24, 21))
        self.browse.setObjectName(_fromUtf8("browse"))

        self.train_btn = QtGui.QPushButton(self.train_tab)
        self.train_btn.setGeometry(QtCore.QRect(20, 110, 291, 27))
        self.train_btn.setObjectName(_fromUtf8("train_btn"))

        self.browse_2 = QtGui.QToolButton(self.train_tab)
        self.browse_2.setGeometry(QtCore.QRect(320, 40, 24, 21))
        self.browse_2.setObjectName(_fromUtf8("browse_2"))

        gui.addTab(self.train_tab, _fromUtf8(""))
        self.search_tab = QtGui.QWidget()
        self.search_tab.setObjectName(_fromUtf8("search_tab"))

        self.searchbox = QtGui.QLineEdit(self.search_tab)
        self.searchbox.setGeometry(QtCore.QRect(10, 10, 251, 23))
        self.searchbox.setObjectName(_fromUtf8("searchbox"))

        self.search_btn = QtGui.QPushButton(self.search_tab)
        self.search_btn.setGeometry(QtCore.QRect(260, 10, 61, 27))
        self.search_btn.setObjectName(_fromUtf8("search_btn"))

        self.toolButton = QtGui.QToolButton(self.search_tab)
        self.toolButton.setGeometry(QtCore.QRect(600, 10, 24, 21))
        self.toolButton.setObjectName(_fromUtf8("browse_3"))

        self.map_path = QtGui.QLineEdit(self.search_tab)
        self.map_path.setGeometry(QtCore.QRect(330, 10, 241, 23))
        self.map_path.setObjectName(_fromUtf8("map_path"))

        gui.addTab(self.search_tab, _fromUtf8(""))

        self.retranslateUi(gui)
        gui.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(gui)

    def retranslateUi(self, gui):
        gui.setWindowTitle(_translate("gui", "SOMSearch", None))
        self.projname_input.setPlaceholderText(_translate("gui", "Project Name", None))
        self.dir_input.setPlaceholderText(_translate("gui", "Html Directory", None))
        self.vecdb_input.setPlaceholderText(_translate("gui", "Vector Database", None))
        self.browse.setText(_translate("gui", "...", None))
        self.train_btn.setText(_translate("gui", "Train", None))
        self.browse_2.setText(_translate("gui", "...", None))
        gui.setTabText(gui.indexOf(self.train_tab), _translate("gui", "Train", None))
        self.searchbox.setPlaceholderText(_translate("gui", "Enter search terms here", None))
        self.search_btn.setText(_translate("gui", "Search", None))
        self.toolButton.setText(_translate("gui", "...", None))
        self.map_path.setPlaceholderText(_translate("gui", "Path to Map", None))
        gui.setTabText(gui.indexOf(self.search_tab), _translate("gui", "Search", None))

        self.train_btn.clicked.connect(self.start_train)
        self.search_btn.clicked.connect(self.start_search)

        QtCore.QObject.connect(self.browse, QtCore.SIGNAL(_fromUtf8("clicked()")), self.browse1)
        QtCore.QObject.connect(self.browse_2, QtCore.SIGNAL(_fromUtf8("clicked()")), self.browsetwo)
        QtCore.QObject.connect(self.toolButton, QtCore.SIGNAL(_fromUtf8("clicked()")), self.browse3)
    def browsetwo(self):
        fname = QFileDialog.getExistingDirectory(self,'Open file')
        self.dir_input.setText(fname)
    def browse1(self):
        fname = QFileDialog.getOpenFileName(self)
        self.vecdb_input.setText(fname)
    def browse3(self):
        fname = QFileDialog.getOpenFileName(self)
        self.map_path.setText(fname)


    def start_train(self):
        pp_v3.get_all_jsons(self.dir_input.text(),self.vecdb_input.text())
        som_v4.train(self.dir_input.text(),self.projname_input.text())

    def start_search(self):
        som_v4.search(self.searchbox.text(),self.map_path.text())


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex=Ui_gui()
    ex.show()
    sys.exit(app.exec_())
