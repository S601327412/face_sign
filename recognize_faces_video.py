
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0


from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--encodings", required=True,
	help="Your encodings lib path")
parser.add_argument("-o", "--output", type=str,
	help="Your video path")
parser.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame")
parser.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model: `hog` or `cnn`,hog is hog algorithm,cnn is Convolution algorithm,"
         "hog more faster and cnn more accurate")
args = parser.parse_args()

print("Read encodings from {}...".format(args.encodings))
data = pickle.loads(open(args.encodings, "rb").read())


print("Open video stream...")
try:
    vs = VideoStream(src=0).start()
except e:
    print("摄像头打开失败:".format(e))
writer = None
time.sleep(2.0)


while True:
    #读取视频流
    start = time.time()
    frame = vs.read()
    # 将BGR图片转为RGB图片
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Resize成(width,height)
    rgb = imutils.resize(frame, width=1024,height=720)
    #缩放比例
    r = frame.shape[1] / float(rgb.shape[1])

    # 得到人脸boxes位置
    boxes = face_recognition.face_locations(rgb,
        model=args.detection_method,number_of_times_to_upsample=1)
    print("识别出{}个人脸".format(len(boxes)))
    # 将图片和人脸位置进行特征编码
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    #遍历编码
    for indx,encoding in enumerate(encodings):
        
        if any(encoding):
        #匹配现有编码库中的编码,小于阈值则为True,大于阈值则为False,默认阈值0.6,返回bool数组
            matches = face_recognition.compare_faces(data["encodings"],
                encoding,tolerance=0.4)

        #未知人脸为Unknown
        name = "Unknown"
        # 找出为True的索引
        if True in matches:
            #索引列表
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # 通过索引找出人脸对应名字
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # 取出索引数最多的那个名字
            name = max(counts, key=counts.get)
        #保存名字
        names.append(name)
        print("第{}个人是{},位置在{}".format(indx+1,names[indx],boxes[indx]))
    # 画框
    for ((top, right, bottom, left), name) in zip(boxes, names):
        #乘以缩放比例
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
        #防止文本信息溢出
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)


    if writer is None and args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output, fourcc, 15,
			(frame.shape[1], frame.shape[0]), True)
    # 写入视频
    if writer is not None:
        writer.write(frame)


    if args.display > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        end = time.time() - start
        print("当前帧耗时{}s".format(end))
        #按q键退出
        if key == ord("q"):
            break

#释放窗口
cv2.destroyAllWindows()
vs.stop()
#释放文件对象
if writer is not None:
	writer.release()