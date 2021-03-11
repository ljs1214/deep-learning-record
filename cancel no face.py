# coding:utf8
import cv2
import dlib
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# 人脸检测的接口，这个是 OpenCV 中自带的
cascade_path = './face_detect_model/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

img_path = "./face_det_img/" # 测试图片路径
images = os.listdir(img_path)
for image in images:
    im = cv2.imread(os.path.join(img_path, image), 1) # 读取图片
    rects = cascade.detectMultiScale(im, 1.3, 5)  # 人脸检测函数
    print("检测到人脸的数量", len(rects))
    if len(rects) == 0:  # len(rects) 是检测人脸的数量，如果没有检测到人脸的话，会显示出图片，适合本地调试使用，在服务器上可能不会显示
        cv2.namedWindow('Result', 0)
        cv2.imshow('Result', im)
        print("没有检测到人脸")
        pass
    plt.imshow(im[:, :, ::-1])  # 显示
    plt.show()
#         os.remove(os.path.join(img_path, image)) #
#         k = cv2.waitKey(0)
#         if k == ord('q'): # 在英文状态下，按下按键 q 会关闭显示窗口
#             break
#     print()
# cv2.destroyAllWindows()
