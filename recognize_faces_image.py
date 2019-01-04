
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

import face_recognition
import argparse
import pickle
import cv2
import time
import imutils

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
parser.add_argument("-i", "--image", required=True,
	help="path to input image")
parser.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`,hog is hog algorithm,cnn is Convolution algorithm,"
         "hog more faster and cnn more accurate")
args = parser.parse_args()


print("Load encodings from {}".format(args.encodings))
data = pickle.loads(open(args.encodings, "rb").read())

start = time.time()
image = cv2.imread(args.image)
#将BGR图片转为RGB图片
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb = imutils.resize(rgb, width=224,height=224)
r = image.shape[1] / float(rgb.shape[1])

print("Starting recognize faces.....")

#得到人脸boxes位置
boxes = face_recognition.face_locations(rgb,
	model=args.detection_method)

#编码
encodings = face_recognition.face_encodings(rgb, boxes)
end = time.time() - start
print("cost:{}".format(end))

names = []

#遍历编码
for encoding in encodings:
    #匹配现有编码库中的编码,小于阈值则为True,大于阈值则为False,返回bool数组
    matches = face_recognition.compare_faces(data["encodings"],
        encoding)
    #未知人脸为Unknown
    name = "Unknown"
    
    #找出为True的索引
    if True in matches:
        #索引列表
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        #通过索引找出人脸对应名字
        for i in matchedIdxs:
	        name = data["names"][i]
	        counts[name] = counts.get(name, 0) + 1

        #取出索引数最多的那个名字
        name = max(counts, key=counts.get)
    #保存名字
    names.append(name)

#画框
for ((top, right, bottom, left), name) in zip(boxes, names):
    top = int(top * r)
    right = int(right * r)
    bottom = int(bottom * r)
    left = int(left * r)
    
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

#可视化
cv2.imshow("Image", image)
cv2.waitKey(0)