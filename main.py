import cv2
import face_recognition
import numpy as np
import datetime
import time
import pickle
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from PyQt5 import QtCore,QtGui
import xlwt
import re

class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture(0)
        self.initUI()
        self.slot_init()
        self.log=[]
        self.count=0
    def initUI(self):

        self.qiandaoButton = QPushButton("签到")
        #qiandaoButton.setGeometry(5,5,10,10)
        self.daochuButton = QPushButton("导出签到日志")
        self.daochuButton.clicked.connect(self.save_file)
        #self.qiandaoButton.clicked.connect(self.show_camera)
        self.text = QTextEdit()
        #self.text.setEnabled(False)
        self.label_display = QLabel("信息:")
        self.label_display.adjustSize()
        self.label_display.setWordWrap(True)
        #daochuButton.setGeometry(10,10,10,10)
        self.image_show =QLabel()
        self.image_show.setFixedSize(641, 481)
        self.image_show.setAutoFillBackground(True)

        wbox = QHBoxLayout()
        hbox = QVBoxLayout()
        #hbox.addStretch(1)
        hbox.addWidget(self.qiandaoButton)
        hbox.addWidget(self.daochuButton)
        hbox.addWidget(self.label_display)
        hbox.addWidget(self.text)
        wbox.addLayout(hbox)
        wbox.addWidget(self.image_show)


        self.setLayout(wbox)
        self.setWindowIcon(QIcon("title.png"))
        self.setGeometry(1000, 1000, 450, 450)
        self.setWindowTitle('人脸识别签到')
        self.show()

    def slot_init(self):

        self.qiandaoButton.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(0)
            if flag == False:
                msg = QMessageBox.warning(self, u"Warning", u"请检测摄像头是否正常",
                                                    buttons=QMessageBox.Ok,
                                                    defaultButton=QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass

            else:
                self.text.setText("请正对着摄像头,开始识别.....")
                self.timer_camera.start(30)


                #self.qiandaoButton.setText(u'签到')

    @pyqtSlot()
    def save_file(self):

        self.filename,self.ok =QFileDialog.getSaveFileName(self,"日志",'/',"All Files (*);;Text Files (*.xlsl)")
        data = xlwt.Workbook(encoding='ascii')
        worksheet = data.add_sheet('My sheet',cell_overwrite_ok=True)
        worksheet.write(0,0,label='姓名')
        worksheet.write(0,1,label='性别')
        worksheet.write(0,2,label='签到时间')
        if self.log:
            for indx,dict in enumerate(self.log):
                    worksheet.write(indx+1,0,label=dict["name"])
                    worksheet.write(indx+1,1,label='男')
                    worksheet.write(indx+1,2,label=dict["data"])
        data.save(self.filename+".xlsx")

    @pyqtSlot()
    def show_camera(self):

        data = pickle.loads(open("encodings.pickle",'rb').read())

        flag,frame = self.cap.read()
        (r,g,b) = cv2.split(frame)
        frame = cv2.merge([b,g,r])

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb,dsize=(640,480))

        r = frame.shape[1]/float(rgb.shape[1])

        boxes = face_recognition.face_locations(rgb,model='cnn', number_of_times_to_upsample=1)

        encodings = face_recognition.face_encodings(rgb,boxes)

        names = []
        dicts ={}

        for indx, encoding in enumerate(encodings):

            if any(encoding):
                # 匹配现有编码库中的编码,小于阈值则为True,大于阈值则为False,默认阈值0.6,返回bool数组
                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding, tolerance=0.4)

            # 未知人脸为Unknown
            name = "Unknown"
            # 找出为True的索引
            if True in matches:
                # 索引列表
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # 通过索引找出人脸对应名字
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # 取出索引数最多的那个名字
                name = max(counts, key=counts.get)
            # 保存名字
            names.append(name)
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # 乘以缩放比例
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 255, 0), 2)
                # 防止文本信息溢出
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
        showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        self.image_show.setPixmap(QtGui.QPixmap.fromImage(showImage))

        if len(names)!=0:
            temp_name =names[0]
            if temp_name=="Unknown":
                self.text.setText("识别失败,未匹配到相应人脸信息:{}".format(temp_name))
                self.count = 0
            elif names[0]==temp_name:

                self.text.setText("识别成功!签到者姓名{},性别:男\n".format(names[0]))
                self.count+= 1
                if self.count==50:
                    #print(temp_name, self.count)
                    self.timer_camera.stop()
                    self.cap.release()
                    QMessageBox.warning(self, u"Finished", u"签到成功",
                                        buttons=QMessageBox.Ok,
                                        defaultButton=QMessageBox.Ok)
                    self.image_show.clear()
                    dicts["name"] = names[0]
                    dicts["data"] = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    self.text.setText("签到成功!签到时间:{}\n".format(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))))
                    #self.text.setLineWrapMode()
                    self.log.append(dicts)
                    self.count=0


    def closeEvent(self, event):
        ok = QPushButton()
        cacel = QPushButton()

        msg = QMessageBox(QMessageBox.Warning, u"退出", u"是否退出！")

        msg.addButton(ok, QMessageBox.ActionRole)
        msg.addButton(cacel, QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')

        if msg.exec_() == QMessageBox.RejectRole:
            event.ignore()
        else:

            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()



if __name__ =="__main__":
    App = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(App.exec_())