
import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from numpy.core.fromnumeric import std

def filter(img,kernel):
    img=np.pad(img,((2,2),(2,2)),'edge')
    output = np.zeros (img.shape)
    for row in range(1, img.shape[0] - 1):
        for col in range(1, img.shape[1] - 1):
            value = kernel * img[(row - 1):(row + 2), (col - 1):(col + 2)]
            output[row-1, col-1] = value.sum ()
    # return output[1:-2,1:-2]
    output = output[1:-2,1:-2]
    # output = (output - output.min())
    # output = output / int(output.max() - output.min())
    # output = output * 255
    # output = np.uint8(output)
    output = 255 - np.abs(output)
    return output

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        
        self.imagepath = ""
        self.img = np.NaN
        self.gray = np.NaN
        self.img_noise = np.NaN
        self.noise = np.NaN
        self.out = np.NaN
        # self.img_num = 0
        # self.noise_std = 0.0

        self.setObjectName("MainWindow")
        self.resize(1030, 642)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setWindowTitle("AIPm10902121")
        
        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setGeometry(QtCore.QRect(10, 10, 70, 20))
        self.btn_load.setObjectName("btn_load")
        self.btn_load.setText("Load")
        self.btn_load.clicked.connect(self.loadfile)
        
        self.conv0_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv0_text.setGeometry(QtCore.QRect(90, 10, 40, 20))
        self.conv0_text.setObjectName("conv0_text")
        self.conv0_text.setEnabled(False)
        self.conv1_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv1_text.setGeometry(QtCore.QRect(140, 10, 40, 20))
        self.conv1_text.setObjectName("conv1_text")
        self.conv1_text.setEnabled(False)
        self.conv2_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv2_text.setGeometry(QtCore.QRect(190, 10, 40, 20))
        self.conv2_text.setObjectName("conv2_text")
        self.conv2_text.setEnabled(False)
        self.conv3_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv3_text.setGeometry(QtCore.QRect(90, 40, 40, 20))
        self.conv3_text.setObjectName("conv3_text")
        self.conv3_text.setEnabled(False)
        self.conv4_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv4_text.setGeometry(QtCore.QRect(140, 40, 40, 20))
        self.conv4_text.setObjectName("conv4_text")
        self.conv4_text.setEnabled(False)
        self.conv5_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv5_text.setGeometry(QtCore.QRect(190, 40, 40, 20))
        self.conv5_text.setObjectName("conv5_text")
        self.conv5_text.setEnabled(False)
        self.conv6_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv6_text.setGeometry(QtCore.QRect(90, 70, 40, 20))
        self.conv6_text.setObjectName("conv6_text")
        self.conv6_text.setEnabled(False)
        self.conv7_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv7_text.setGeometry(QtCore.QRect(140, 70, 40, 20))
        self.conv7_text.setObjectName("conv7_text")
        self.conv7_text.setEnabled(False)
        self.conv8_text = QtWidgets.QTextEdit(self.centralwidget)
        self.conv8_text.setGeometry(QtCore.QRect(190, 70, 40, 20))
        self.conv8_text.setObjectName("conv7_text")
        self.conv8_text.setEnabled(False)

        self.btn_set = QtWidgets.QPushButton(self.centralwidget)
        self.btn_set.setGeometry(QtCore.QRect(90, 100, 70, 20))
        self.btn_set.setObjectName("btn_set")
        self.btn_set.setText("Set")
        self.btn_set.setEnabled(False)
        self.btn_set.clicked.connect(self.set)

        self.btn_conv = QtWidgets.QPushButton(self.centralwidget)
        self.btn_conv.setGeometry(QtCore.QRect(240, 10, 70, 20))
        self.btn_conv.setObjectName("btn_conv")
        self.btn_conv.setText("Conv.")
        self.btn_conv.setEnabled(False)
        self.btn_conv.clicked.connect(self.conv)
        
        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setGeometry(QtCore.QRect(320, 10, 70, 20))
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setText("Save")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.savefile)
       
        self.btn_exit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_exit.setGeometry(QtCore.QRect(400, 10, 70, 20))
        self.btn_exit.setObjectName("btn_exit")
        self.btn_exit.setText("Exit")
        self.btn_exit.clicked.connect(self.close)        
        
        self.input_label = QtWidgets.QLabel(self.centralwidget)
        self.input_label.setGeometry(QtCore.QRect(10, 120, 47, 12))
        self.input_label.setText("Input:")
        self.input_label.setObjectName("input_label")

        self.output_label = QtWidgets.QLabel(self.centralwidget)
        self.output_label.setGeometry(QtCore.QRect(520, 120, 47, 12))
        self.output_label.setText("Output:")
        self.output_label.setObjectName("output_label")

        self.input_pic = QtWidgets.QLabel(self.centralwidget)
        self.input_pic.setGeometry(QtCore.QRect(10, 132, 500, 500))
        self.input_pic.setText("")
        self.input_pic.setObjectName("input_pic")

        self.output_pic = QtWidgets.QLabel(self.centralwidget)
        self.output_pic.setGeometry(QtCore.QRect(520, 132, 500, 500))
        self.output_pic.setText("")
        self.output_pic.setObjectName("output_pic")

        self.setCentralWidget(self.centralwidget)
        

        
    def loadfile(self):
        
        filePath = QtWidgets.QFileDialog.getOpenFileName(self)
        self.imagepath = filePath[0]
        try:
            self.img = np.NaN
            self.img = cv.imread(self.imagepath, cv.IMREAD_GRAYSCALE)
            cv.imwrite("gray.jpg", self.img)                
            self.input_pic.setPixmap(QtGui.QPixmap('gray.jpg'))
            os.remove('gray.jpg')
            self.conv0_text.setEnabled(True)
            self.conv1_text.setEnabled(True)
            self.conv2_text.setEnabled(True)
            self.conv3_text.setEnabled(True)
            self.conv4_text.setEnabled(True)
            self.conv5_text.setEnabled(True)
            self.conv6_text.setEnabled(True)
            self.conv7_text.setEnabled(True)
            self.conv8_text.setEnabled(True)
            self.btn_set.setEnabled(True)
            self.btn_conv.setEnabled(False)
            
            # self.btn_setstd.setEnabled(True)
            # self.std_text.setEnabled(True)
        except cv.error as e:
            return        
        # self.btn_save.setEnabled(True)
        
    def savefile(self):                  
        save_path = os.path.splitext(self.imagepath)[0]
        cv.imwrite(save_path+'_conv.jpg', self.out)        
        self.btn_save.setEnabled(True)        
        

    def set(self):
        self.conv0_text.setEnabled(False)
        self.conv1_text.setEnabled(False)
        self.conv2_text.setEnabled(False)
        self.conv3_text.setEnabled(False)
        self.conv4_text.setEnabled(False)
        self.conv5_text.setEnabled(False)
        self.conv6_text.setEnabled(False)
        self.conv7_text.setEnabled(False)
        self.conv8_text.setEnabled(False)
        self.btn_conv.setEnabled(True)
        
    def conv(self):
        self.btn_save.setEnabled(True)
        try:
            conv0 = float(self.conv0_text.toPlainText())
            conv1 = float(self.conv1_text.toPlainText())
            conv2 = float(self.conv2_text.toPlainText())
            conv3 = float(self.conv3_text.toPlainText())
            conv4 = float(self.conv4_text.toPlainText())
            conv5 = float(self.conv5_text.toPlainText())
            conv6 = float(self.conv6_text.toPlainText())
            conv7 = float(self.conv7_text.toPlainText())
            conv8 = float(self.conv8_text.toPlainText())
        except ValueError as e:            
            conv0 = 1
            conv1 = 1
            conv2 = 1
            conv3 = 1
            conv4 = 1
            conv5 = 1
            conv6 = 1
            conv7 = 1
            conv8 = 1    
        
        kernel = np.array([
        [conv0, conv1, conv2], 
        [conv3, conv4, conv5], 
        [conv6, conv7, conv8]])
        self.out = filter(self.img, kernel)
        cv.imwrite('out.jpg', self.out)
        self.output_pic.setPixmap(QtGui.QPixmap('out.jpg'))
        os.remove('out.jpg')

        self.conv0_text.setEnabled(True)
        self.conv1_text.setEnabled(True)
        self.conv2_text.setEnabled(True)
        self.conv3_text.setEnabled(True)
        self.conv4_text.setEnabled(True)
        self.conv5_text.setEnabled(True)
        self.conv6_text.setEnabled(True)
        self.conv7_text.setEnabled(True)
        self.conv8_text.setEnabled(True)
        self.btn_conv.setEnabled(False)         

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = MyWindow()
    ui.show()
    sys.exit(app.exec_())
