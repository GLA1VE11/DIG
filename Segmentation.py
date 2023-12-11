# -*- coding = utf-8 -*-
# @Time : 2021-12-10 10:46
# @Author: gla1ve
# @File: Segmentation.py
# @Software: PyCharm
# 人生苦短， 我用python(划掉) Java
import math

import cv2
import numpy as np
import warnings


class Segmentation:
    # cv2.imread第二个参数是0的时候表示读取灰度图。
    def __init__(self, pic_path):
        self.pic_path = pic_path
        self.pic = cv2.imread(self.pic_path, 0).astype(np.uint8)
        self.HH = self.pic.shape[0] - 1
        self.WW = self.pic.shape[1] - 1

    def ShowPic(self):
        cv2.imshow('PKQ', self.pic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字，毫秒级

    # 基于灰度直方图的阈值分割
    def Gray_Histogram_Segmentation(self):
        output = np.zeros(self.pic.shape, np.uint8)
        for i in range(1, self.pic.shape[0] - 1):
            for j in range(1, self.pic.shape[1] - 1):
                if self.pic[i][j] <= 100:
                    output[i][j] = 0
                else:
                    output[i][j] = 255
        cv2.imwrite('img/Segmentation.bmp', output)
        cv2.imshow('Gray_Histogram_Segmentation', output)
        cv2.waitKey(0)

    # 基于类内方差最小值的分割(最小类内方差法，也叫作均匀性度量法)
    def Minimum_Variance(self):
        pic_data = self.pic
        # 遍历0到255 的所有阈值分割的条件【这里可以二分嘛？】
        res, Variance_min = 0, 1e9
        for th in range(0, 250, 30):
            left = [pic_data[i][j] for i in range(self.HH) for j in range(self.WW) if pic_data[i][j] >= th]
            right = [pic_data[i][j] for i in range(self.HH) for j in range(self.WW) if pic_data[i][j] < th]
            tmp = (len(left)) / (len(left) + len(right)) * np.var(left) + \
                  (len(right)) / (len(left) + len(right)) * np.var(right)
            if tmp < Variance_min:
                Variance_min, res = tmp, th
        for i in range(self.HH):
            for j in range(self.WW):
                if pic_data[i][j] <= res:
                    pic_data[i][j] = 0
                else:
                    pic_data[i][j] = 255
        cv2.imwrite('img/Segmentation.bmp', pic_data)
        cv2.imshow('Minimum_Variance', pic_data)
        cv2.waitKey(0)

    # p 参数生长
    def P_Parameter(self):
        pic_data = np.zeros((self.HH, self.WW), np.uint8)
        img = self.pic
        th, p = 127, 0.2
        left = [img[x][y] for x in range(self.HH) for y in range(self.WW) if img[x][y] < th]
        ps = len(left) / self.HH / self.WW
        while math.fabs(ps - p) > 0.04:
            if ps < p:
                th += 1
            else:
                th -= 1
            if ps == 255:
                break
            left = [img[x][y] for x in range(self.HH) for y in range(self.WW) if img[x][y] < th]
            ps = len(left) / self.HH / self.WW
        for i in range(self.HH):
            for j in range(self.WW):
                if img[i][j] <= th:
                    pic_data[i][j] = 0
                else:
                    pic_data[i][j] = 255
        # print(th)
        cv2.imwrite('img/Segmentation.bmp', pic_data)
        cv2.imshow('P_Parameter', pic_data)
        cv2.waitKey(0)

    # 腐蚀
    def Erosion(self):
        pic_data = np.zeros((self.HH, self.WW), np.uint8)
        img = self.pic
        for i in range(self.HH):
            for j in range(self.WW):
                if math.fabs(img[i][j] - 60) <= 60:
                    pic_data[i][j] = 0
                else:
                    pic_data[i][j] = 255

        for i in range(self.HH - 1):
            for j in range(self.WW - 1):
                if pic_data[i][j] == 0 and pic_data[i + 1][j] == 0 and pic_data[i][j + 1] == 0 \
                        and pic_data[i + 1][j + 1] == 0:
                    img[i][j] = 0
                else:
                    img[i][j] = 255
        cv2.imwrite('img/Segmentation.bmp', img)
        cv2.imshow('Erosion', img)
        cv2.waitKey(0)

    # 膨胀
    def Inflation(self):
        pic_data = np.zeros((self.HH, self.WW), np.uint8)
        img = self.pic
        for i in range(self.HH):
            for j in range(self.WW):
                if math.fabs(img[i][j] - 60) <= 60:
                    pic_data[i][j] = 0
                else:
                    pic_data[i][j] = 255

        for i in range(self.HH - 1):
            for j in range(self.WW - 1):
                if pic_data[i][j] == 255 and pic_data[i + 1][j] == 255 and pic_data[i][j + 1] == 255 and \
                        pic_data[i + 1][j + 1] == 255:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
        cv2.imwrite('img/Segmentation.bmp', img)
        cv2.imshow('Inflation', img)
        cv2.waitKey(0)

    # 闭运算
    def Close_Operation(self):
        pic_data = np.zeros((self.HH, self.WW), np.uint8)
        img = self.pic
        for i in range(self.HH):
            for j in range(self.WW):
                if math.fabs(img[i][j] - 60) <= 60:
                    pic_data[i][j] = 0
                else:
                    pic_data[i][j] = 255

        for i in range(self.HH - 1):
            for j in range(self.WW - 1):
                if pic_data[i][j] == 0 and pic_data[i + 1][j] == 0 and pic_data[i][j + 1] == 0 \
                        and pic_data[i + 1][j + 1] == 0:
                    img[i][j] = 0
                else:
                    img[i][j] = 255

        for i in range(self.HH - 1):
            for j in range(self.WW - 1):
                if pic_data[i][j] == 255 and pic_data[i + 1][j] == 255 and pic_data[i][j + 1] == 255 and \
                        pic_data[i + 1][j + 1] == 255:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
        cv2.imwrite('img/Segmentation.bmp', img)
        cv2.imshow('Close_Operation', img)
        cv2.waitKey(0)

    # 开运算
    def Open_Operation(self):
        pic_data = np.zeros((self.HH, self.WW), np.uint8)
        img = self.pic
        for i in range(self.HH):
            for j in range(self.WW):
                if math.fabs(img[i][j] - 60) <= 60:
                    pic_data[i][j] = 0
                else:
                    pic_data[i][j] = 255

        for i in range(self.HH - 1):
            for j in range(self.WW - 1):
                if pic_data[i][j] == 255 and pic_data[i + 1][j] == 255 and pic_data[i][j + 1] == 255 and \
                        pic_data[i + 1][j + 1] == 255:
                    img[i][j] = 255
                else:
                    img[i][j] = 0

        for i in range(self.HH - 1):
            for j in range(self.WW - 1):
                if pic_data[i][j] == 0 and pic_data[i + 1][j] == 0 and pic_data[i][j + 1] == 0 \
                        and pic_data[i + 1][j + 1] == 0:
                    img[i][j] = 0
                else:
                    img[i][j] = 255

        cv2.imwrite('img/Segmentation.bmp', img)
        cv2.imshow('Open_Operation', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    path = 'img/build.bmp'
    b = Segmentation(path)
    b.Open_Operation()
