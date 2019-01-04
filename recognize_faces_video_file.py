
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4 --output output/lunch_scene_output.avi --display 0


import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--encodings", required=True,
	help="Your encodings lib path")
parser.add_argument("-i", "--input", required=True,
	help="Your test video input")
parser.add_argument("-o", "--output", type=str,
	help="path to output video")
parser.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
parser.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`,hog is hog algorithm,cnn is Convolution algorithm,"
         "hog more faster and cnn more accurate")
args = parser.parse_args()

print("Loading encodings from {}".format(args.encodings))
data = pickle.loads(open(args.encodings, "rb").read())

#VideoCapture 参数为0代表打开摄像头,若给文件路径,则打开视频文件
if args.input:
    print("Open video file from {}....".format(args.input))
    try:
        stream = cv2.VideoCapture(args.input)
    except e:
        print("打开文件失败:".format(e))
else:
    print("Open camera.....")
    try:
        stream = cv2.VideoCapture(0)
    except e:
        print("摄像头打开失败:".format(e))
writer = None
count=0
while True:
    #读取视频流
    start = time.time()
    (grabbed, frame) = stream.read()
    #最后一帧退出
    #print(grabbed)
    if not grabbed:
        break
    #将BGR图像转换为RGB图像
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #resize为640X480
    rgb = imutils.resize(frame, width=640,height=480)
    #缩放比例
    r = frame.shape[1] / float(rgb.shape[1])
    # 得到人脸boxes位置
    boxes = face_recognition.face_locations(rgb,
        model=args.detection_method,number_of_times_to_upsample=1)
    print("boxes:{}".format(boxes))
    # 将图片和人脸位置进行特征编码
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    names = []
    # 遍历特征编码
    for encoding in encodings:
        # 匹配现有编码库中的编码,小于阈值则为True,大于阈值则为False,返回bool数组
        matches = face_recognition.compare_faces(data["encodings"],
			encoding)
        #未知人脸为Unknown
        
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
            
        #保存名字
        names.append(name)
        print("Name:{}".format(names))
    #画框
    for ((top, right, bottom, left), name) in zip(boxes, names):
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
        ourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output, ourcc, 20,
			(640, 480), True)
    
    # 写入视频
    if writer is not None:
        writer.write(frame)

    #是否展示识别效果
    
    if args.display > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        end = time.time()-start
        count = count + 1
        print("cost_{}s".format(end))
        # 按q键退出
        if key == ord("q"):
            break

#关闭流
stream.release()
#关闭文件对象
if writer is not None:
	writer.release()