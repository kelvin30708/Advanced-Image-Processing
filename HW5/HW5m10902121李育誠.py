
import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
# from numpy.core.fromnumeric import std
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        #parameters
        self.imagepath = ""
        self.img = np.NaN
        self.gray = np.NaN
        self.gray_eq = np.NaN
        self.H = []
        self.H_eq = []
        self.x = np.arange(0,256)
        # self.noise = np.NaN
        # self.img_num = 0
        # self.noise_std = 0.0
        
        # ui configuration
        self.setObjectName("MainWindow")
        self.resize(1050, 930)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setWindowTitle("AIPm10902121")
        
        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setGeometry(QtCore.QRect(10, 10, 70, 20))
        self.btn_load.setObjectName("btn_load")
        self.btn_load.setText("Load")
        self.btn_load.clicked.connect(self.loadfile)

        self.btn_his_eq = QtWidgets.QPushButton(self.centralwidget)
        self.btn_his_eq.setGeometry(QtCore.QRect(110, 10, 70, 20))
        self.btn_his_eq.setObjectName("btn_his_eq")
        self.btn_his_eq.setText("his_eq")
        self.btn_his_eq.clicked.connect(self.his_eq)
        self.btn_his_eq.setEnabled(False)

        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setGeometry(QtCore.QRect(190, 10, 70, 20))
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setText("Save")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.savefile)
    
        self.btn_exit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_exit.setGeometry(QtCore.QRect(270, 10, 70, 20))
        self.btn_exit.setObjectName("btn_exit")
        self.btn_exit.setText("Exit")
        self.btn_exit.clicked.connect(self.close)
        
        

        
        
        self.input_label = QtWidgets.QLabel(self.centralwidget)
        self.input_label.setGeometry(QtCore.QRect(10, 50, 47, 12))
        self.input_label.setText("Input:")
        self.input_label.setObjectName("input_label")

        self.his_eq_label = QtWidgets.QLabel(self.centralwidget)
        self.his_eq_label.setGeometry(QtCore.QRect(10, 472, 120, 12))
        self.his_eq_label.setText("Histogram Equalization:")
        self.his_eq_label.setObjectName("his_eq_label")

        # self.his_eq_label = QtWidgets.QLabel(self.centralwidget)
        # self.his_eq_label.setGeometry(QtCore.QRect(10, 472, 120, 12))
        # self.his_eq_label.setText("Histogram Equalization:")
        # self.his_eq_label.setObjectName("his_eq_label")

        self.histogram_label = QtWidgets.QLabel(self.centralwidget)
        self.histogram_label.setGeometry(QtCore.QRect(520, 50, 200, 12))
        self.histogram_label.setText("histogram:")
        self.histogram_label.setObjectName("histogram_label")

        self.out_histogram_label = QtWidgets.QLabel(self.centralwidget)
        self.out_histogram_label.setGeometry(QtCore.QRect(520, 472, 200, 12))
        self.out_histogram_label.setText("Eq histogram:")
        self.out_histogram_label.setObjectName("histogram_label")

        self.input_pic = QtWidgets.QLabel(self.centralwidget)
        self.input_pic.setGeometry(QtCore.QRect(10, 62, 500, 400))
        self.input_pic.setText("")
        self.input_pic.setObjectName("input_pic")

        self.output_pic = QtWidgets.QLabel(self.centralwidget)
        self.output_pic.setGeometry(QtCore.QRect(10, 484, 500, 400))
        self.output_pic.setText("")
        self.output_pic.setObjectName("output_pic")

        self.histogram_pic = QtWidgets.QLabel(self.centralwidget)
        self.histogram_pic.setGeometry(QtCore.QRect(520, 62, 500, 400))
        self.histogram_pic.setText("")
        self.histogram_pic.setObjectName("histogram_pic")

        self.output_histogram_pic = QtWidgets.QLabel(self.centralwidget)
        self.output_histogram_pic.setGeometry(QtCore.QRect(520, 484, 500, 400))
        self.output_histogram_pic.setText("")
        self.output_histogram_pic.setObjectName("output_pic")

        self.setCentralWidget(self.centralwidget)
        

        
    def loadfile(self):
        
        filePath = QtWidgets.QFileDialog.getOpenFileName(self)
        self.imagepath = filePath[0]
        try:
            self.img = np.NaN
            self.gray = np.NaN
            self.img = cv.imread(self.imagepath)
            self.gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
            self.img = self.gray
            cv.imwrite("gray.jpg", self.gray)                
            self.input_pic.setPixmap(QtGui.QPixmap('gray.jpg'))
            os.remove('gray.jpg')            
            self.btn_his_eq.setEnabled(True)

            hist = cv.calcHist([self.gray], [0], None, [256], [0, 256])
            self.H = []
            for x in hist:
                self.H.append(x[0])
            x = np.arange(0,256)
            plt.figure()
            plt.bar(x, self.H)
            plt.savefig(self.imagepath + "tmp.jpg")
            tmp = cv.imread(self.imagepath + "tmp.jpg")
            tmp = cv.resize(tmp, (500, 400), interpolation=cv.INTER_AREA)
            cv.imwrite(self.imagepath + "tmp.jpg", tmp)
            self.histogram_pic.setPixmap(QtGui.QPixmap(self.imagepath + "tmp.jpg"))
            os.remove(self.imagepath + "tmp.jpg")
        except cv.error as e:
            return
       
    def savefile(self):                  
        save_path = os.path.splitext(self.imagepath)[0]
        # cv.imwrite('gray_eq.jpg', self.gray_eq)
        cv.imwrite(save_path+ "_histogram_eq_pic.jpg", self.gray_eq)
        # print(self.gray_eq)
        x = np.arange(0,256)
        plt.figure()
        plt.bar(x, self.H)
        plt.savefig(save_path+ "_histogram.jpg")
        plt.close()
        
        plt.figure()
        plt.bar(x, self.H_eq)
        plt.savefig(save_path+ "_histogram_eq.jpg")     
        plt.close()
        
        self.btn_save.setEnabled(True)        
        

    def his_eq(self):
        m, n = self.gray.shape        
        # initial cultimative function
        Hc = []
        Hc.append(self.H[0])

        # iterate cultimative function
        for g in range(1, 256):
            Hc.append(Hc[g-1] + self.H[g])
        
        # histogram equalization algotithm implementation
        Hmin = min(self.H)
        T = []
        for g in range(0, 256):
            T.append(round(((Hc[g] - Hmin)/(m*n - Hmin))*255))
        
        #Rescan the image and write an output image with gray-levels gq, setting gq = T[gp] 
        self.gray_eq = np.zeros(shape=(m, n), dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                gp = self.gray[i][j]
                self.gray_eq[i][j] = T[gp]
        cv.imwrite('gray_eq.jpg', self.gray_eq)
        self.output_pic.setPixmap(QtGui.QPixmap("gray_eq.jpg"))
        os.remove("gray_eq.jpg")

        hist = cv.calcHist([self.gray_eq], [0], None, [256], [0, 256])
        self.H_eq = []
        for x in hist:
            self.H_eq.append(x[0])
        x = np.arange(0,256)
        plt.figure()
        plt.bar(x, self.H_eq)
        plt.savefig(self.imagepath + "tmp.jpg")
        tmp = cv.imread(self.imagepath + "tmp.jpg")
        tmp = cv.resize(tmp, (500, 400), interpolation=cv.INTER_AREA)
        cv.imwrite(self.imagepath + "tmp.jpg", tmp)
        self.output_histogram_pic.setPixmap(QtGui.QPixmap(self.imagepath + "tmp.jpg"))
        # self.histogram_pic.setPixmap(QtGui.QPixmap(self.imagepath + "tmp.jpg"))
        os.remove(self.imagepath + "tmp.jpg")
        self.btn_his_eq.setEnabled(False)
        self.btn_save.setEnabled(True)






        
        # plt.figure()
        # hist = cv.calcHist([self.gray_eq], [0], None, [256], [0, 256])
        #     self.H = []
        #     for x in hist:
        #         self.H.append(x[0])
        # QtGui.QImage()
        
        # plt.savefig('tmp.jpg')
        # self.output_pic.setPixmap(QtGui.QPixmap('histogram.jpg'))
        # print(type(QtGui.QPixmap('histogram.jpg')))
        # shave histogram
        # self.histogram_pic.setPixmap(QtGui.QPixmap('tmp.jpg'))

        # self.btn_save.setEnabled(True)        
        # plt.close()
        # os.remove('tmp.jpg')

# def his_eq(gray):
#     m, n = gray.shape
#     # to calculate the histogram    
#     hist = cv.calcHist([gray], [0], None, [256], [0, 256])
#     H = []
#     for x in hist:
#         H.append(x[0])
#     x = np.arange(0,256)
    
#     # initial cultimative function
#     Hc = []
#     Hc.append(H[0])

#     # iterate cultimative function
#     for g in range(1, 256):
#         Hc.append(Hc[g-1] + H[g])
    
#     # histogram equalization algotithm implementation
#     Hmin = min(H)
#     T = []
#     for g in range(0, 256):
#         T.append(round(((Hc[g] - Hmin)/(m*n - Hmin))*255))
    
#     #Rescan the image and write an output image with gray-levels gq, setting gq = T[gp] 
#     gray_eq = np.zeros(shape=(m, n), dtype=np.uint8)
#     for i in range(m):
#         for j in range(n):
#             gp = gray[i][j]
#             gray_eq[i][j] = T[gp]
#     return gray_eq



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = MyWindow()
    ui.show()
    sys.exit(app.exec_())
