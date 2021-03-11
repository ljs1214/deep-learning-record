# coding:utf8

import sys
import numpy as np
import cv2
import os
import dlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
from PIL import Image
import torch.nn.functional as F

import matplotlib.pyplot as plt
import warnings
import tensorflow as tf


warnings.filterwarnings('ignore')

PREDICTOR_PATH = "./face_detect_model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path = './face_detect_model/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

if not os.path.exists("results"):
    os.mkdir("results")

class simpleconv3(nn.Module):
    def __init__(self):
        super(simpleconv3,self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 5 * 5 , 1200)
        self.fc2 = nn.Linear(1200 , 128)
        self.fc3 = nn.Linear(128 , 4)

    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        #print "bn1 shape",x.shape
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 48 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3, 5)
    x, y, w, h = rects[0]
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im,
                    str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


testsize = 48  # 测试图大小

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
net = simpleconv3()
net.eval()
modelpath = "./models/model.ckpt"  # 模型路径
net.load_state_dict(
    torch.load(modelpath, map_location=lambda storage, loc: storage))

# 一次测试一个文件
img_path = "./test_img/"
imagepaths = os.listdir(img_path)  # 图像文件夹
for imagepath in imagepaths:
    im = cv2.imread(os.path.join(img_path, imagepath), 1)
    try:
        rects = cascade.detectMultiScale(im, 1.3, 5)
        x, y, w, h = rects[0]
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = np.matrix([[p.x, p.y]
                               for p in predictor(im, rect).parts()])
    except:
        print("没有检测到人脸")
        continue  # 没有检测到人脸

    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0

    for i in range(48, 67):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y

    roiwidth = xmax - xmin
    roiheight = ymax - ymin

    roi = im[ymin:ymax, xmin:xmax, 0:3]

    if roiwidth > roiheight:
        dstlen = 1.5 * roiwidth
    else:
        dstlen = 1.5 * roiheight

    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight

    newx = xmin
    newy = ymin

    imagerows, imagecols, channel = im.shape
    if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
        newx = newx - diff_xlen / 2
    elif newx < diff_xlen / 2:
        newx = 0
    else:
        newx = imagecols - dstlen

    if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
        newy = newy - diff_ylen / 2
    elif newy < diff_ylen / 2:
        newy = 0
    else:
        newy = imagecols - dstlen

    roi = im[int(newy):int(newy + dstlen), int(newx):int(newx + dstlen), 0:3]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roiresized = cv2.resize(roi,
                            (testsize, testsize)).astype(np.float32) / 255.0
    imgblob = data_transforms(roiresized).unsqueeze(0)
    imgblob.requires_grad = False
    imgblob = Variable(imgblob)
    torch.no_grad()
    predict = F.softmax(net(imgblob))
    print(predict)
    index = np.argmax(predict.detach().numpy())

    im_show = cv2.imread(os.path.join(img_path, imagepath), 1)
    im_h, im_w, im_c = im_show.shape
    pos_x = int(newx + dstlen)
    pos_y = int(newy + dstlen)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(im_show, (int(newx), int(newy)),
                  (int(newx + dstlen), int(newy + dstlen)), (0, 255, 255), 2)
    if index == 0:
        cv2.putText(im_show, 'none', (pos_x, pos_y), font, 1.5, (0, 0, 255), 2)
    if index == 1:
        cv2.putText(im_show, 'pout', (pos_x, pos_y), font, 1.5, (0, 0, 255), 2)
    if index == 2:
        cv2.putText(im_show, 'smile', (pos_x, pos_y), font, 1.5, (0, 0, 255), 2)
    if index == 3:
        cv2.putText(im_show, 'open', (pos_x, pos_y), font, 1.5, (0, 0, 255), 2)
    #     cv2.namedWindow('result', 0)
    #     cv2.imshow('result', im_show)
    cv2.imwrite(os.path.join('results', imagepath), im_show)
    #     print(os.path.join('results', imagepath))
    plt.imshow(im_show[:, :, ::-1])  # 这里需要交换通道，因为 matplotlib 保存图片的通道顺序是 RGB，而在 OpenCV 中是 BGR
    plt.show()
#     cv2.waitKey(0)
# cv2.destroyAllWindows()