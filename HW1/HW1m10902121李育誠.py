# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2 as cv
import numpy as np
# class MyWindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super(MyWindow, self).__init__()
#         self.initUI        
        
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        
        self.imagepath = ""
        self.img = np.NaN
        
        
        self.setObjectName("MainWindow")
        self.resize(1030, 572)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setWindowTitle("AIPm10902121")
        
        self.btn_browse = QtWidgets.QPushButton(self.centralwidget)
        self.btn_browse.setGeometry(QtCore.QRect(10, 10, 70, 20))
        self.btn_browse.setObjectName("btn_browse")
        self.btn_browse.setText("Browse")
        self.btn_browse.clicked.connect(self.browsefile)

        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setGeometry(QtCore.QRect(90, 10, 70, 20))
        self.btn_load.setObjectName("btn_load")
        self.btn_load.setText("Do sth.")
        self.btn_load.setEnabled(False)

        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setGeometry(QtCore.QRect(170, 10, 70, 20))
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setText("Save")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.savefile)

        self.btn_exit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_exit.setGeometry(QtCore.QRect(250, 10, 70, 20))
        self.btn_exit.setObjectName("btn_exit")
        self.btn_exit.setText("Exit")
        self.btn_exit.clicked.connect(self.close)

        self.input_label = QtWidgets.QLabel(self.centralwidget)
        self.input_label.setGeometry(QtCore.QRect(10, 50, 47, 12))
        self.input_label.setText("Input:")
        self.input_label.setObjectName("input_label")

        self.output_label = QtWidgets.QLabel(self.centralwidget)
        self.output_label.setGeometry(QtCore.QRect(520, 50, 47, 12))
        self.output_label.setText("Output:")
        self.output_label.setObjectName("output_label")

        self.input_pic = QtWidgets.QLabel(self.centralwidget)
        self.input_pic.setGeometry(QtCore.QRect(10, 62, 500, 500))
        self.input_pic.setText("")
        # self.input_pic.setPixmap(QtGui.QPixmap("C:/Users/Li/Desktop/3660496_1.jpg"))
        self.input_pic.setObjectName("input_pic")

        self.output_pic = QtWidgets.QLabel(self.centralwidget)
        self.output_pic.setGeometry(QtCore.QRect(520, 62, 500, 500))
        self.output_pic.setText("")
        # self.output_pic.setPixmap(QtGui.QPixmap("C:/Users/Li/Desktop/3660496_1.jpg"))
        self.output_pic.setObjectName("output_pic")
        self.setCentralWidget(self.centralwidget)
        

        
    def browsefile(self):
        filePath = QtWidgets.QFileDialog.getOpenFileName(self)
        self.imagepath = filePath[0]
        
        self.input_pic.setPixmap(QtGui.QPixmap(self.imagepath))
        self.output_pic.setPixmap(QtGui.QPixmap(self.imagepath))
        self.btn_save.setEnabled(True)
        # print(type(filePath[0]))
        
    def savefile(self):
        img = cv.imread(self.imagepath)          
        cv.imwrite("out.bmp", img)
        self.btn_save.setEnabled(False)
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = MyWindow()
    ui.show()
    sys.exit(app.exec_())
