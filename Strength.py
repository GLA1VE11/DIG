# -*- coding = utf-8 -*-
# @Time : 2021-11-01 10:57
# @Author: gla1ve
# @File: Strength.py.py
# @Software: PyCharm
# 人生苦短， 我用python(划掉) Java
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import MultipleLocator
from heapq import nsmallest
from BMP import BMP  # 前一个BMP是文件名，后一个BMP是类名

'''
upd: 2021-11-13
我认为我很多函数写的很乱，有的是直接修改self.pic，而有的是修改了tmp, 只是用cv2显示出来了而已
'''


class Strength:
    def __init__(self, pic_path):
        self.pic_path = pic_path
        self.pic = cv2.imread(self.pic_path, cv2.IMREAD_COLOR).astype(np.uint8)

    def ShowPic(self):
        cv2.imshow('PKQ', self.pic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字，毫秒级

    # 分三个通道绘制R,G,B的灰度直方图  这里bmp=1只是一个占位符,  如果flag是True,bmp会传入一个BMP类的对象
    def ShowHist(self, bmp=1, flag=False):  # 里面有些plt的用法，值得积累qwq
        if flag is False:
            bmp = BMP(self.pic_path)
        blue, green, red = bmp.ToArray()
        B = np.bincount(blue)
        G = np.bincount(green)
        R = np.bincount(red)
        scale = MultipleLocator(50)
        Bins = 150
        # ax为两条坐标轴的实例
        plt.rcParams['figure.figsize'] = (8, 10)
        plt.subplots_adjust(wspace=0.2, hspace=0.5)  # 设置子图的间隔
        # 1 / 8
        plt.subplot(421), plt.hist(blue, bins=Bins, density=True)
        plt.title('Blue'), plt.ylabel("ratio")
        # 2 / 8
        plt.subplot(422), plt.plot(np.arange(B.shape[0]), B, 'b')
        plt.xlim(0, 256), plt.title('Blue')
        ax = plt.gca()
        ax.xaxis.set_major_locator(scale)
        # 3 / 8
        plt.subplot(423), plt.hist(green, bins=Bins, density=True)
        plt.title('Green'), plt.ylabel("ratio")
        # 4 / 8
        plt.subplot(424), plt.plot(np.arange(G.shape[0]), G, 'g')
        plt.xlim(0, 256), plt.title('Green')
        ax = plt.gca()
        ax.xaxis.set_major_locator(scale)
        # 5 / 8
        plt.subplot(425), plt.hist(red, bins=Bins, density=True)
        plt.title('Red'), plt.ylabel("ratio")
        # 6 / 8
        plt.subplot(426), plt.plot(np.arange(R.shape[0]), R, 'r')
        plt.xlim(0, 256), plt.title('Red')
        ax = plt.gca()
        ax.xaxis.set_major_locator(scale)
        # 7 / 8
        mix = (blue / 3 + green / 3 + red / 3).astype(np.uint8)
        M = np.bincount(mix)
        plt.subplot(427), plt.hist(mix, bins=Bins, density=True)
        plt.title('Avg'), plt.ylabel("ratio")
        # 8 / 8
        plt.subplot(428), plt.plot(np.arange(M.shape[0]), M, 'b')
        plt.xlim(0, 256), plt.title('Avg')
        ax = plt.gca()
        ax.xaxis.set_major_locator(scale)
        plt.show()

    # 直方图均衡化 (仅限真彩图, 三个通道分开均衡化)
    def Histogram_Equalization2(self):
        bmp = BMP(self.pic_path)
        # self.ShowHist(bmp=bmp, flag=True)
        assert (bmp.IsRealColor is True)
        blue, green, red = bmp.ToArray()
        B = np.bincount(blue)
        G = np.bincount(green)
        R = np.bincount(red)
        b_arr, g_arr, r_arr = np.zeros(256), np.zeros(256), np.zeros(256)
        # 归一化后求前缀和
        for i in range(B.shape[0]):
            b_arr[i] = B[i]
        for i in range(G.shape[0]):
            g_arr[i] = G[i]
        for i in range(R.shape[0]):
            r_arr[i] = R[i]
        for i in range(1, 256):
            b_arr[i] += b_arr[i - 1]
            g_arr[i] += g_arr[i - 1]
            r_arr[i] += r_arr[i - 1]
        b_arr = b_arr / bmp.biHeight / bmp.biWidth
        b_arr = (b_arr * 255 + 0.5).astype(np.uint8)
        g_arr = g_arr / bmp.biHeight / bmp.biWidth
        g_arr = (g_arr * 255 + 0.5).astype(np.uint8)
        r_arr = r_arr / bmp.biHeight / bmp.biWidth
        r_arr = (r_arr * 255 + 0.5).astype(np.uint8)
        for i in range(bmp.biHeight):
            for j in range(bmp.biWidth):
                bmp.data[i][j][0] = b_arr[int(bmp.data[i][j][0] + 0.5)]
                bmp.data[i][j][1] = g_arr[int(bmp.data[i][j][1] + 0.5)]
                bmp.data[i][j][2] = r_arr[int(bmp.data[i][j][2] + 0.5)]
        # self.ShowHist(bmp=bmp, flag=True)
        cv2.imwrite('img/Strength.bmp', bmp.data[:, :].astype(np.uint8))
        cv2.imshow('After Histogram_Equalization', bmp.data[:, :].astype(np.uint8))
        cv2.waitKey(0)

    # 直方图均衡化 (仅限真彩图, 三个通道一起均衡化) 这个不太准，应该用下面那个分开的
    def Histogram_Equalization(self):
        bmp = BMP(self.pic_path)
        # self.ShowHist(bmp=bmp, flag=True)
        assert (bmp.IsRealColor is True)
        blue, green, red = bmp.ToArray()
        mix = (blue / 3 + green / 3 + red / 3 + 0.5).astype(np.uint8)
        tmp = np.bincount(mix)
        M = np.zeros(256)
        # 归一化后求前缀和
        for i in range(tmp.shape[0]):
            M[i] = tmp[i]
        for i in range(1, 256):
            M[i] += M[i - 1]
        M = M / bmp.biHeight / bmp.biWidth
        M = (M * 255 + 0.5).astype(np.uint8)
        for i in range(bmp.biHeight):
            for j in range(bmp.biWidth):
                for k in range(3):
                    bmp.data[i][j][k] = M[int(bmp.data[i][j][k] + 0.5)]
        # self.ShowHist(bmp=bmp, flag=True)
        # bmp.SaveBMP()
        # bmp.ShowPic(BMP.default_save_path)
        # plt.imshow(bmp.data / 255)
        # plt.show()
        # print(bmp.data[:5])
        """
        上面的方法都不用了 终于摸索出来了。。 plt.imshow的时候：
        要么强制转换成uint8, 要么就用浮点数进行/255归一化；
        而cv2.imshow传数组的时候，会把RGB和在一起，所以bmp处理完之后，只能取前2维
        """
        cv2.imwrite('img/Strength.bmp', bmp.data[:, :].astype(np.uint8))
        cv2.imshow('After Histogram_Equalization', bmp.data[:, :].astype(np.uint8))
        cv2.waitKey(0)

    # 显示二值化后的图片 (这里只能用幻数了, 建议先Hist看一下应该设置成几qwq)
    '''
    python一切皆引用。 这里Img会直接修改self.pic， 所以使用二值化函数之后，原来的对象会变成黑白的
    '''
    def Binarization(self):
        Img = self.pic
        for i in range(Img.shape[0]):
            for j in range(Img.shape[1]):
                v = Img[i][j]
                if any(v <= 100):       # 幻数
                    v = 0
                else:
                    v = 255
                Img[i][j] = v
        cv2.imwrite('img/Strength.bmp', Img)
        cv2.imshow('Binarization', Img)
        cv2.waitKey(0)

    # 显示灰度级切片后的图片， 对原对象进行修改
    def Grey_Cutter(self):
        Img = self.pic
        for i in range(Img.shape[0]):
            for j in range(Img.shape[1]):
                v = Img[i][j]
                if any(v >= 120):  # 幻数
                    v = 255
                else:
                    v = 0
                Img[i][j] = v
        cv2.imwrite('img/Strength.bmp', Img)
        cv2.imshow('Binarization', Img)
        cv2.waitKey(0)

    # 显示灰度化后的图片
    def Grey_Pic(self):
        Img = cv2.imread(self.pic_path, 0).astype(np.uint8)
        cv2.imwrite('img/Strength.bmp', Img)
        cv2.imshow('Grey', Img)
        cv2.waitKey(0)

    # 为图片添加随机噪声
    '''
    注意，我这里对图片添加椒盐、随机、高斯噪声，都不是对原对象进行修改，而是利用cv2将图片保存到某个地方。
    '''
    def Generate_Random_Noise(self, prob=0.02):
        output = np.zeros(self.pic.shape, np.uint8)
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = self.pic[i][j] + random.random() * 255
                else:
                    output[i][j] = self.pic[i][j]
        cv2.imwrite('img/Noised.bmp', output)
        cv2.imshow('Salt_And_Pepper_Noise', output)
        cv2.waitKey(0)

    # 为图片添加椒盐噪声
    def Generate_Salt_And_Pepper_Noise(self, prob=0.02):
        output = np.zeros(self.pic.shape, np.uint8)
        upb = 1 - prob
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > upb:
                    output[i][j] = 255
                else:
                    output[i][j] = self.pic[i][j]
        # output = np.asarray(output).astype(np.uint8)
        cv2.imwrite('img/Noised.bmp', output)
        cv2.imshow('Salt_And_Pepper_Noise', output)
        cv2.waitKey(0)

    # 为图片添加高斯噪声
    '''
    np.clip()用法：
    整个数组的值限制在指定值a_min与a_max之间，对比a_min小的和比a_max大的值就重置为a_min,和a_max。
    对于每个位置添加gauss噪声， 然后显示出来。
    '''
    def Generate_Gauss_Noise(self, mean=0, var=0.001):
        image = np.array(self.pic / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        cv2.imwrite('img/Noised.bmp', out)
        cv2.imshow('Gauss_Noise', out)
        cv2.waitKey(0)

    # 高斯滤波[高斯模糊](调库实现的 cv2)
    def Gaussian_Blur(self, n=3):
        out = cv2.GaussianBlur(self.pic, (n, n), 0)
        cv2.imwrite('img/Strength.bmp', out)
        cv2.imshow('Gauss_Noise', out)
        cv2.waitKey(0)

    # 均值滤波
    '''
    注意这里的tmp是[0, 0, 0], 所以取均值的时候直接/8即可, 而不是np.average[这样会求出一个数作为均值，然后广播给img。这会
    导致RGB三个通道的值是一样的，就变成灰图了]
    注意在使用np.average median的时候, 要结合axis来使用。 否则就是对所有数字求了
    '''
    def Average_Filtering(self):
        img = np.zeros(self.pic.shape, np.uint8)
        dir = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        # 等价于: dir = [[x, y] for x in range(-1, 2) for y in range(-1, 2)]
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                if 0 < i < self.pic.shape[0] - 1 and 0 < j < self.pic.shape[1] - 1:
                    tmp = np.zeros_like(self.pic.shape)
                    for k in range(8):
                        tmp += self.pic[i + dir[k][0]][j + dir[k][1]]
                    img[i][j] = (tmp / 8).astype(np.uint8)
                else:
                    img[i][j] = self.pic[i][j].astype(np.uint8)
        cv2.imwrite('img/Strength.bmp', img)
        cv2.imshow('Average_Filtering', img)
        cv2.waitKey(0)

    # 中值滤波
    def Median_Filtering(self):
        img = np.zeros(self.pic.shape, np.uint8)
        dir = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                if 0 < i < self.pic.shape[0] - 1 and 0 < j < self.pic.shape[1] - 1:
                    tmp = []
                    for k in range(8):
                        tmp.append(self.pic[i + dir[k][0]][j + dir[k][1]])
                    img[i][j] = np.median(tmp, axis=0).astype(np.uint8)
                else:
                    img[i][j] = self.pic[i][j].astype(np.uint8)
        cv2.imwrite('img/Strength.bmp', img)
        cv2.imshow('Median_Filtering', img)
        cv2.waitKey(0)

    # SNN, 对称近邻均值滤波, 取一个（2n+1)*(2n+1)的区域, 这里默认取的是5*5的。算一下，大约是512*512*5*5的.非常慢
    '''
    这里有一个问题： 对于真彩图像RGB三个通道来说，是分开看哪个离得近呢，还是合起来看哪个离得近？
    e.g. 目标点[50, 50, 50], d1 = [200, 50, 50], d2 = [50, 200, 200], 是分开变成[50, 50, 50]，还是直接选d1?
    '''
    def SNN_Filtering(self, n=2):
        N = n * 2 + 1
        img = np.zeros(self.pic.shape, np.uint8)
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                if n <= i < self.pic.shape[0] - n and n <= j < self.pic.shape[1] - n:
                    tmp = np.zeros_like(self.pic.shape)
                    # 接下来对每一个点处理
                    for x in range(i - n, i + 1):
                        for y in range(j - n, j + n + 1):
                            if x == i and y == j:
                                break
                            d1, d2 = self.pic[x][y], self.pic[2 * i - x][2 * j - y]
                            for zz in range(3):
                                # print(d1[zz], d2[zz], self.pic[i][j][zz])
                                if np.abs(int(d1[zz]) - self.pic[i][j][zz]) < np.abs(int(d2[zz]) - self.pic[i][j][zz]):
                                    tmp[zz] += d1[zz]
                                else:
                                    tmp[zz] += d2[zz]
                            # exit(0)
                            # if np.abs(np.sum(d1 - self.pic[i][j])) < np.abs(np.sum(d2 - self.pic[i][j])):
                            #     tmp += d1
                            # else:
                            #     tmp += d2
                    img[i][j] = (tmp / (N * N // 2)).astype(np.uint8)
                else:
                    img[i][j] = self.pic[i][j].astype(np.uint8)
        cv2.imwrite('img/Strength.bmp', img)
        cv2.imshow('SNN_Filtering', img)
        cv2.waitKey(0)

    # K-均值滤波 这里使用的是N = 5, K = 7
    def KNN_Filtering(self):
        dir = [[x, y] for x in range(-2, 3) for y in range(-2, 3)]
        img = np.zeros(self.pic.shape, np.uint8)
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                if 2 <= i < self.pic.shape[0] - 2 and 2 <= j < self.pic.shape[1] - 2:
                    R, G, B = [], [], []
                    for k in range(25):
                        B.append(self.pic[i + dir[k][0]][j + dir[k][1]][0])
                        G.append(self.pic[i + dir[k][0]][j + dir[k][1]][1])
                        R.append(self.pic[i + dir[k][0]][j + dir[k][1]][2])
                    b = nsmallest(7, B, key=lambda x: abs(int(x) - self.pic[i][j][0]))
                    g = nsmallest(7, G, key=lambda x: abs(int(x) - self.pic[i][j][1]))
                    r = nsmallest(7, R, key=lambda x: abs(int(x) - self.pic[i][j][2]))
                    img[i][j] = np.asarray([np.average(b), np.average(g), np.average(r)]).astype(np.uint8)
                else:
                    img[i][j] = self.pic[i][j]
        cv2.imwrite('img/Strength.bmp', img)
        cv2.imshow('KNN_Filtering', img)
        cv2.waitKey(0)

    # 二值图像的黑白点噪声滤波
    def Black_And_White_Dot_Noise_Filtering(self):
        dir = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                if 0 < i < self.pic.shape[0] - 1 and 0 < j < self.pic.shape[1] - 1:
                    tmp = np.zeros_like(self.pic.shape)
                    for k in range(8):
                        tmp += self.pic[i + dir[k][0]][j + dir[k][1]]
                    if any(np.abs(self.pic[i][j] - np.average(tmp / 8)) >= 127.5):
                        self.pic[i][j] = 255 - self.pic[i][j]
        cv2.imwrite('img/Strength.bmp', self.pic)
        cv2.imshow('Black_And_White_Dot_Noise_Filtering', self.pic)
        cv2.waitKey(0)


if __name__ == '__main__':
    # from heapq import nsmallest
    # s = [[1, 2, 3, 4, 5, 6, 7], [2, 4, 5, 6, 8, 86]]
    # print(nsmallest(3, s, key=lambda x: abs(x - 50)))
    # exit(0)
    save_path = r'E:\大三上学期课内\数字图像\图像处理图片\Lena\tmp\save.bmp'
    path = 'img/Snow_blur.bmp'
    b = Strength(path)
    b.Grey_Pic()
    # b.Histogram_Equalization2()
    # b.Binarization()
    # b.Grey_Cutter()
    # b.Gaussian_Blur()
    # b.Generate_Salt_And_Pepper_Noise()
    # b.KNN_Filtering()
    # b.Generate_Random_Noise(prob=0.5)
    # b.Grey_Cutter()
    # b.Generate_Gauss_Noise(var=0.01)
    # b.SNN_Filtering(n=2)
    # b.Black_And_White_Dot_Noise_Filtering()
    # b.Median_Filtering()
    # b.Average_Filtering()
    # b.Grey_Cutter()
    # b.ShowHist()
    # b.Grey_Cutter()
    # hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])
    # print(hist_cv)
    # plt.subplot(121), plt.imshow(img, 'gray')
    # plt.subplot(122), plt.hist(hist_cv, density=True)
    # plt.show()
    # a = Strength(path)
    # a.ShowPic()
