# -*- coding = utf-8 -*-
# @Time : 2021-11-30 14:59
# @Author: gla1ve
# @File: Sharpen.py
# @Software: PyCharm
# 人生苦短， 我用python(划掉) Java
import cv2
import numpy as np


class Sharpen:
    # cv2.imread第二个参数是0的时候表示读取灰度图。锐化都是对灰度图而言的，所以初始化函数直接把图片读取成灰度图。
    def __init__(self, pic_path):
        self.pic_path = pic_path
        self.pic = cv2.imread(self.pic_path, 0).astype(np.uint8)

    def ShowPic(self):
        cv2.imshow('PKQ', self.pic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字，毫秒级

    # 水平方向的一阶锐化【采用直接取绝对值的方法】
    def Horizontal_Sharpen(self):
        output = np.zeros(self.pic.shape, np.uint8)
        for i in range(1, self.pic.shape[0] - 1):
            for j in range(1, self.pic.shape[1] - 1):
                output[i][j] = np.abs(self.pic[i - 1][j - 1] + 2 * self.pic[i - 1][j] + self.pic[i - 1][j + 1]
                                      - self.pic[i + 1][j - 1] - 2 * self.pic[i + 1][j] - self.pic[i + 1][j + 1])
        cv2.imwrite('img/Sharpen.bmp', output)
        cv2.imshow('Horizontal_Sharpen', output)
        cv2.waitKey(0)

    # 垂直方向的一阶锐化【采用直接取绝对值的方法】
    def Vertical_Sharpen(self):
        output = np.zeros(self.pic.shape, np.uint8)
        for i in range(1, self.pic.shape[0] - 1):
            for j in range(1, self.pic.shape[1] - 1):
                output[i][j] = np.abs(self.pic[i - 1][j - 1] + 2 * self.pic[i][j - 1] + self.pic[i + 1][j - 1]
                                      - self.pic[i - 1][j + 1] - 2 * self.pic[i][j + 1] - self.pic[i + 1][j + 1])
        cv2.imwrite('img/Sharpen.bmp', output)
        cv2.imshow('Vertical_Sharpen', output)
        cv2.waitKey(0)

    '''
    基于门限值的梯度锐化:
    比较像素的梯度是否大于30，是则将梯度值加100，不是则将该像素点的灰度值恢复，如果梯度加160大于255，将其置为255；
    '''
    def Gradient_Sharpen(self):
        output = np.zeros(self.pic.shape, np.uint8)
        for i in range(1, self.pic.shape[0] - 1):
            for j in range(1, self.pic.shape[1] - 1):
                t1 = np.abs(int(self.pic[i][j]) - int(self.pic[i - 1][j]))
                t2 = np.abs(int(self.pic[i][j]) - int(self.pic[i][j - 1]))
                tmp = t1 + t2
                if tmp > 30:
                    tmp += 100
                tmp = 255 if tmp + 60 > 255 else tmp
                output[i][j] = tmp
        cv2.imwrite('img/Sharpen.bmp', output)
        cv2.imshow('Gradient_Sharpen', output)
        cv2.waitKey(0)

    # Robert算子
    def Robert_Sharpen(self):
        r, c = self.pic.shape[0], self.pic.shape[1]
        Robert = [[-1, -1], [1, 1]]
        for x in range(r):
            for y in range(c):
                if y + 2 <= c and x + 2 <= r:
                    imgChild = self.pic[x:x + 2, y:y + 2]
                    list_robert = Robert * imgChild
                    self.pic[x, y] = abs(list_robert.sum())  # 求和加绝对值
        cv2.imwrite('img/Sharpen.bmp', self.pic)
        cv2.imshow('Robert_Sharpen', self.pic)
        cv2.waitKey(0)

    # Sobel算子
    def Sobel_Sharpen(self):
        r, c = self.pic.shape[0], self.pic.shape[1]
        new_image = np.zeros((r, c), np.uint8)
        new_imageX = np.zeros_like(self.pic).astype(np.uint)
        new_imageY = np.zeros_like(self.pic).astype(np.uint)
        SobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        SobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        for i in range(1, r - 1):
            for j in range(1, c - 1):
                new_imageX[i, j] = abs(np.sum(self.pic[i - 1:i + 2, j - 1:j + 2] * SobelX))
                new_imageY[i, j] = abs(np.sum(self.pic[i - 1:i + 2, j - 1:j + 2] * SobelY))
                new_image[i, j] = (new_imageX[i, j] * new_imageX[i, j] + new_imageY[
                    i, j] * new_imageY[i, j]) ** 0.5
        cv2.imwrite('img/Sharpen.bmp', new_image)
        cv2.imshow('Sobel_Sharpen', new_image)
        cv2.waitKey(0)

    # Prewitt算子
    def Prewitt_Sharpen(self):
        r, c = self.pic.shape[0], self.pic.shape[1]
        new_image = np.zeros((r, c), np.uint8)
        new_imageX = np.zeros_like(self.pic).astype(np.uint)
        new_imageY = np.zeros_like(self.pic).astype(np.uint)
        PrewittX = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        PrewittY = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        for i in range(1, r - 1):
            for j in range(1, c - 1):
                new_imageX[i, j] = abs(np.sum(self.pic[i - 1:i + 2, j - 1:j + 2] * PrewittX))
                new_imageY[i, j] = abs(np.sum(self.pic[i - 1:i + 2, j - 1:j + 2] * PrewittY))
                new_image[i, j] = (new_imageX[i, j] * new_imageX[i, j] + new_imageY[
                    i, j] * new_imageY[i, j]) ** 0.5
        cv2.imwrite('img/Sharpen.bmp', new_image)
        cv2.imshow('Prewitt_Sharpen', new_image)
        cv2.waitKey(0)

    # Laplacian算子
    def Laplacian_Sharpen(self):
        r, c = self.pic.shape[0], self.pic.shape[1]
        Laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        new_image = np.zeros((r, c), np.uint8)
        for i in range(1, r - 1):
            for j in range(1, c - 1):
                new_image[i, j] = abs(np.sum(self.pic[i - 1: i + 2, j - 1: j + 2] * Laplacian))
        cv2.imwrite('img/Sharpen.bmp', new_image)
        cv2.imshow('Laplacian_Sharpen', new_image)
        cv2.waitKey(0)

    # 最接近原图的Laplacian算子
    def Laplacian2_Sharpen(self):
        r, c = self.pic.shape[0], self.pic.shape[1]
        Laplacian = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        new_image = np.zeros((r, c), np.uint8)
        for i in range(1, r - 1):
            for j in range(1, c - 1):
                new_image[i, j] = abs(np.sum(self.pic[i - 1: i + 2, j - 1: j + 2] * Laplacian))
        cv2.imwrite('img/Sharpen.bmp', new_image)
        cv2.imshow('Laplacian2_Sharpen', new_image)
        cv2.waitKey(0)

    # Wallis算子
    def Wallis_Sharpen(self):
        r, c = self.pic.shape[0], self.pic.shape[1]
        tmp = 46 * np.log(self.pic + 1e-5).astype(np.uint8)
        Wallis = np.array([[0, -1.0 / 4, 0], [-1.0 / 4, 1, -1.0 / 4], [0, -1.0 / 4, 0]])
        new_image = np.zeros((r, c))
        for i in range(1, r - 1):
            for j in range(1, c - 1):
                new_image[i, j] = np.abs(np.sum(tmp[i - 1: i + 2, j - 1: j + 2] * Wallis))
        cv2.imwrite('img/Sharpen.bmp', new_image)
        cv2.imshow('Wallis_Sharpen', new_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    path = 'img/build.bmp'
    b = Sharpen(path)
    b.Wallis_Sharpen()
