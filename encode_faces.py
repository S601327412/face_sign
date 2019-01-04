# python encode_faces.py --dataset dataset --encodings encodings.pickle

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--dataset", required=True,
	help="Your dataset path")
parser.add_argument("-e", "--encodings", required=True,
	help="Your enconding's lib path")
parser.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model: `hog` or `cnn`,hog is hog algorithm,cnn is Convolution algorithm,"
         "hog more faster and cnn more accurate")
args = parser.parse_args()

# 获取数据集路径
print("Get dataset.....from {}".format(args.dataset))
imagePaths = list(paths.list_images(args.dataset))

# 初始化编码列表和名字字符串
knownEncodings = []
knownNames = []

# 遍历数据集
for (i, imagePath) in enumerate(imagePaths):
	print("processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# 获取人脸位置
	boxes = face_recognition.face_locations(rgb,
		model=args.detection_method)

	#将图片和人脸位置进行特征编码
	encodings = face_recognition.face_encodings(rgb, boxes)
    #编码列表和名字标签列表
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

#将编码列表和名字标签列表写入本地
print("Write encoding to local file ")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args.encodings, "wb")
f.write(pickle.dumps(data))
f.close()