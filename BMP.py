# -*- coding = utf-8 -*-
# @Time : 2021-10-15 14:04
# @Author: gla1ve
# @File: BMP.py
# @Software: PyCharm
# 人生苦短， 我用python(划掉) Java
import math
import numpy as np
import struct
import matplotlib.pyplot as plt

'''
2021-12-01 BMP类作为一个内嵌类进行了修改, 原本在控制台输出的东西转到QT中
'''

class BMP:
    # 这个de路径是默认的保存路径，不要改成原图的路径， 否则会覆盖掉
    default_save_path = 'img/bmp_save.bmp'

    # 读取位图信息
    def __init__(self, pic_path):
        # 读取位图文件头
        self.path = pic_path
        file = open(pic_path, "rb")
        self.bfType, = struct.unpack("<h", file.read(2))
        if self.bfType != 19778:
            print("文件不是BMP位图！")
            return  # 不是BM, 直接返回
        self.isBMP = True
        self.bfSize, = struct.unpack("<i", file.read(4))
        self.bfReserved1, = struct.unpack("<h", file.read(2))
        self.bfReserved2, = struct.unpack("<h", file.read(2))
        self.bfOffBits, = struct.unpack("<i", file.read(4))

        # 读取位图信息头
        self.biSize, = struct.unpack("<i", file.read(4))  # 该结构的长度 为40
        self.biWidth, = struct.unpack("<i", file.read(4))  # 图像的宽度
        self.biHeight, = struct.unpack("<i", file.read(4))  # 图像的高度
        self.biPlanes, = struct.unpack("<h", file.read(2))  # 平面数 必须为1
        self.biBitCount, = struct.unpack("<h", file.read(2))  # 颜色位数
        self.biCompression, = struct.unpack("<i", file.read(4))  # 图像是否压缩
        self.biSizeImage, = struct.unpack("<i", file.read(4))  # 图像大小 以字节为单位
        self.biXPelsPerMeter, = struct.unpack("<i", file.read(4))  # 水平分辨率
        self.biYPelsPerMeter, = struct.unpack("<i", file.read(4))  # 垂直分辨率
        self.biClrUsed, = struct.unpack("<i", file.read(4))  # 实际使用的彩色表中的颜色索引数
        self.biClrImportant, = struct.unpack("<i", file.read(4))  # 对图像显示有重要影响的颜色索引的数目
        # 读取调色板(这里设置只有颜色位数<=8才有调色板)
        if self.biBitCount >= 16:
            self.HasAlpha = True if self.biBitCount == 32 else False
            self.RGBQUAD = np.zeros(0)
            self.IsRealColor = True
        else:
            self.RGBQUAD = np.zeros((1 << self.biBitCount, 4))
            self.IsRealColor = False
            for i in range(1 << self.biBitCount):
                self.RGBQUAD[i][0], = struct.unpack("<B", file.read(1))
                self.RGBQUAD[i][1], = struct.unpack("<B", file.read(1))
                self.RGBQUAD[i][2], = struct.unpack("<B", file.read(1))
                self.RGBQUAD[i][3], = struct.unpack("<B", file.read(1))

        # 读取位图数据
        if not self.IsRealColor:
            self.data = np.zeros((self.biHeight, self.biWidth))
            for i in range(self.biHeight):
                for j in range(self.biWidth):
                    index, = struct.unpack("<B", file.read(1))
                    self.data[self.biHeight - 1 - i][j] = index
        else:
            if not self.HasAlpha:
                self.data = np.zeros((self.biHeight, self.biWidth, 3))
                for i in range(self.biHeight):
                    for j in range(self.biWidth):
                        for k in range(3):
                            index, = struct.unpack("<B", file.read(1))
                            self.data[self.biHeight - 1 - i][j][k] = index
            else:
                self.data = np.zeros((self.biHeight, self.biWidth, 4))
                for i in range(self.biHeight):
                    for j in range(self.biWidth):
                        for k in range(4):
                            index, = struct.unpack("<B", file.read(1))
                            self.data[self.biHeight - 1 - i][j][k] = index
        file.close()

    # 显示图片（使用ImShow方法）
    @staticmethod
    def ShowPic(pic_path):
        pic = plt.imread(pic_path)
        plt.imshow(pic)
        plt.axis('off')
        plt.show()

    # 显示图片信息
    def ShowPicInfo(self, flag=False):
        info = ""
        info += f"该图片是BMP位图\n图像宽度为{self.biWidth}, 高度为{self.biHeight}\n"
        info += f"该图使用{self.biBitCount}位描述颜色\n"
        if self.IsRealColor:
            info += "该图是真彩图\n"
        else:
            info += f"该图不是真彩图,是{1 << self.biBitCount}色图像\n"
        if flag:
            self.ShowPic(self.path)
        info += f"图像实际使用的的颜色索引数:{self.biClrUsed}\n"
        return info

    # 将图片信息以数组的形式返回（这样真彩和非真彩的图片就没有区别了）
    def ToArray(self):
        if self.IsRealColor:
            blue = (self.data[:, :, 0].flatten() + 0.5).astype(np.uint8)
            green = (self.data[:, :, 1].flatten() + 0.5).astype(np.uint8)
            red = (self.data[:, :, 2].flatten() + 0.5).astype(np.uint8)
            return blue, green, red
        else:       # 非真彩图直接把调色板中的数据放进去即可
            blue = np.array([self.RGBQUAD[int(x)][0] for x in self.data.flatten()]).astype(np.uint8)
            green = np.array([self.RGBQUAD[int(x)][1] for x in self.data.flatten()]).astype(np.uint8)
            red = np.array([self.RGBQUAD[int(x)][2] for x in self.data.flatten()]).astype(np.uint8)
            return blue, green, red

    # 保存图片, 有默认路径
    def SaveBMP(self, save_path=default_save_path):
        file = open(save_path, "wb+")
        width = self.biWidth  # 保存下,原始的宽
        # 对宽向上取整，保证四字节对齐(如果这里不对齐，是无法读出的)
        self.biWidth = ((self.biWidth + 3) // 4) * 4
        # 计算变化后的所需字节数 和图像大小
        self.bfSize = 14 + 40 + self.RGBQUAD.size // 4 + int(self.biHeight * self.biWidth / 4)
        self.biSizeImage = int(self.biHeight * self.biWidth / 4)
        # 写入文件头
        file.write(struct.pack("H", self.bfType))
        file.write(struct.pack("I", self.bfSize))
        file.write(struct.pack("H", self.bfReserved1))
        file.write(struct.pack("H", self.bfReserved2))
        file.write(struct.pack("I", self.bfOffBits))
        # reconstruct bmp header
        file.write(struct.pack("I", self.biSize))
        file.write(struct.pack("I", self.biWidth))
        file.write(struct.pack("I", self.biHeight))
        file.write(struct.pack("H", self.biPlanes))
        file.write(struct.pack("H", self.biBitCount))
        file.write(struct.pack("I", self.biCompression))
        file.write(struct.pack("I", self.biSizeImage))
        file.write(struct.pack("I", self.biXPelsPerMeter))
        file.write(struct.pack("I", self.biYPelsPerMeter))
        file.write(struct.pack("I", self.biClrUsed))
        file.write(struct.pack("I", self.biClrImportant))

        for i in range(self.RGBQUAD.size // 4):
            file.write(struct.pack("<B", int(self.RGBQUAD[i][0])))
            file.write(struct.pack("<B", int(self.RGBQUAD[i][1])))
            file.write(struct.pack("<B", int(self.RGBQUAD[i][2])))
            file.write(struct.pack("<B", int(self.RGBQUAD[i][3])))

        if not self.IsRealColor:
            for i in range(self.biHeight):
                for j in range(self.biWidth):
                    if j < width:
                        file.write(struct.pack("<B", int(self.data[self.biHeight - 1 - i][j])))
                    else:
                        file.write(struct.pack("<B", 0))
        else:
            for i in range(self.biHeight):
                for j in range(self.biWidth):
                    for k in range(self.data.shape[-1]):
                        if j < width:
                            file.write(struct.pack("<B", int(self.data[self.biHeight - 1 - i][j][k])))
                        else:
                            file.write(struct.pack("<B", 0))
        file.close()

    # 移动图片，改变图片的大小，原始图片完全保留
    def Move(self, x, y, save_path=default_save_path):
        h_tmp = int(self.biHeight + abs(x) + 0.5)
        w_tmp = int(self.biWidth + abs(y) + 0.5)
        if self.IsRealColor:
            new_data = np.zeros((h_tmp, w_tmp, self.data.shape[-1]), np.uint8)
        else:
            new_data = np.zeros((h_tmp, w_tmp), np.uint8)
        for i in range(self.biHeight):
            for j in range(self.biWidth):
                i0 = int(i + x + 0.5) if x > 0 else i
                j0 = int(j + y + 0.5) if y > 0 else j
                new_data[i0][j0] = self.data[i][j]
        self.data = new_data
        # print(self.data.shape)
        self.biHeight = h_tmp
        self.biWidth = w_tmp
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 移动图片，不改变图片的大小，原始图片部分消失
    def Move_Without_Changing_Size(self, x, y, save_path=default_save_path):
        new_data = np.zeros_like(self.data, np.uint8)
        for i in range(self.biHeight):
            for j in range(self.biWidth):
                i0 = int(i - x + 0.5)
                j0 = int(j - y + 0.5)
                if 0 <= i0 < self.biHeight and 0 <= j0 < self.biWidth:
                    new_data[i][j] = self.data[i0][j0]
        self.data = new_data
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 水平镜像
    def Mirror_Horizontally(self, save_path=default_save_path):
        new_data = np.zeros_like(self.data, np.uint8)
        for i in range(self.biHeight):
            for j in range(self.biWidth):
                new_data[i][j] = self.data[i][self.biWidth - j - 1]
        self.data = new_data
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 垂直镜像
    def Mirror_Vertically(self, save_path=default_save_path):
        new_data = np.zeros_like(self.data, np.uint8)
        for i in range(self.biHeight):
            for j in range(self.biWidth):
                new_data[i][j] = self.data[self.biHeight - i - 1][j]
        self.data = new_data
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 中心对称
    def Central_symmetry(self, save_path=default_save_path):
        new_data = np.zeros_like(self.data, np.uint8)
        for i in range(self.biHeight):
            for j in range(self.biWidth):
                new_data[i][j] = self.data[self.biHeight - i - 1][self.biWidth - j - 1]
        self.data = new_data
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    '''
    图像逆时针旋转,插值(这里定义counterclockwise方向为正), 旋转中心是图像中心. 注意屏幕坐标的x, y是反着的
    过于辣鸡，狗都不用 不但慢，效果不好，还tm有bug
    Q:有一说一，非真彩图可以直接插值就很离谱，但是就算把RGB直接算出来，也没法hash到调色板qwq
    A:2021/10/20 upd: 是我孤陋寡闻了 调色板存储的颜色是渐变的 所以貌似可以这样。 不过效果一如既往地拉胯 
    '''

    def Rotate_Counterclockwise_Interpolation(self, angle, save_path=default_save_path):
        angle = angle * math.pi / 180  # 角度弧度转化
        centerY, centerX = self.biWidth / 2, self.biHeight / 2
        tmpx, tmpy = [], []
        # 旋转后的图像的像素的最大值和最小值仍然在四个顶点处取得
        for x0, y0 in ((0, 0), (0, self.biHeight - 1), (self.biWidth - 1, 0), (self.biHeight - 1, self.biWidth - 1)):
            x = (x0 - centerX) * math.cos(angle) - (y0 - centerY) * math.sin(angle) + centerX
            y = (x0 - centerX) * math.sin(angle) + (y0 - centerY) * math.cos(angle) + centerY
            tmpx.append(int(x + 0.5)), tmpy.append(int(y + 0.5))
        # 获得新图的长宽, 并声明新图区域
        new_height, new_width = int(max(tmpx) - min(tmpx)) + 1, int(max(tmpy) - min(tmpy)) + 1
        if not self.IsRealColor:
            new_data = np.zeros((new_height, new_width), np.uint8)
        else:
            new_data = np.zeros((new_height, new_width, self.data.shape[-1]), np.uint8)
        # 开始旋转
        for i in range(self.biHeight):
            for j in range(self.biWidth):
                i0 = int((i - centerX) * math.cos(angle) - (j - centerY) * math.sin(angle) + centerX - min(tmpx))
                j0 = int((i - centerX) * math.sin(angle) + (j - centerY) * math.cos(angle) + centerY - min(tmpy))
                new_data[i0][j0] = self.data[i][j]
        # 进行八邻域的均值插值填充
        # dir = ((-1, 0), (0, -1), (0, 1), (1, 0))  四邻域用
        if not self.IsRealColor:
            dir = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
            for i in range(new_height):
                for j in range(new_width):
                    if new_data[i][j] == 0:
                        num, sum = 0, 0
                        for x0, y0 in dir:
                            tx, ty = i + x0, j + y0
                            if 0 <= tx < new_height and 0 <= ty < new_width:
                                num = num + 1
                                sum = sum + new_data[tx][ty]
                        sum = sum // num if num > 0 else sum
                        new_data[i][j] = sum
        else:
            dir = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
            for i in range(new_height):
                for j in range(new_width):
                    if new_data[i][j][0] == 0 and new_data[i][j][1] == 0 and new_data[i][j][2] == 0:
                        blue, green, red = [], [], []
                        for x0, y0 in dir:
                            tx, ty = i + x0, j + y0
                            if 0 <= tx < new_height and 0 <= ty < new_width:
                                blue.append(new_data[tx][ty][0])
                                green.append(new_data[tx][ty][1])
                                red.append(new_data[tx][ty][2])
                        new_data[i][j][0] = np.average(blue)
                        new_data[i][j][1] = np.average(green)
                        new_data[i][j][2] = np.average(red)
        self.data = new_data
        self.biHeight = new_height
        self.biWidth = new_width
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    '''
    图像顺时针旋转，插值(这里定义clockwise方向为正), 旋转中心是图像中心. 注意屏幕坐标的x, y是反着的
    上面那个辣鸡的孪生兄弟，狗都不用
    '''

    def Rotate_Clockwise_Interpolation(self, angle, save_path=default_save_path):
        angle = angle * math.pi / 180  # 角度弧度转化
        centerY, centerX = self.biWidth / 2, self.biHeight / 2
        tmpx, tmpy = [], []
        # 旋转后的图像的像素的最大值和最小值仍然在四个顶点处取得
        for x0, y0 in ((0, 0), (0, self.biHeight - 1), (self.biWidth - 1, 0), (self.biHeight - 1, self.biWidth - 1)):
            x = (x0 - centerX) * math.cos(angle) + (y0 - centerY) * math.sin(angle) + centerX
            y = -(x0 - centerX) * math.sin(angle) + (y0 - centerY) * math.cos(angle) + centerY
            # print(x, y)
            tmpx.append(int(x + 0.5)), tmpy.append(int(y + 0.5))
        # 获得新图的长宽, 并声明新图区域
        new_height, new_width = int(max(tmpx) - min(tmpx)) + 1, int(max(tmpy) - min(tmpy)) + 1
        if not self.IsRealColor:
            new_data = np.zeros((new_height, new_width), np.uint8)
        else:
            new_data = np.zeros((new_height, new_width, self.data.shape[-1]), np.uint8)
        # 开始旋转
        for i in range(self.biHeight):
            for j in range(self.biWidth):
                i0 = int((i - centerX) * math.cos(angle) + (j - centerY) * math.sin(angle) + centerX - min(tmpx))
                j0 = int(-(i - centerX) * math.sin(angle) + (j - centerY) * math.cos(angle) + centerY - min(tmpy))
                new_data[i0][j0] = self.data[i][j]
        # 进行八邻域的均值插值填充
        # dir = ((-1, 0), (0, -1), (0, 1), (1, 0))  四邻域用
        if not self.IsRealColor:
            dir = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
            for i in range(new_height):
                for j in range(new_width):
                    if new_data[i][j] == 0:
                        num, sum = 0, 0
                        for x0, y0 in dir:
                            tx, ty = i + x0, j + y0
                            if 0 <= tx < new_height and 0 <= ty < new_width:
                                num = num + 1
                                sum = sum + new_data[tx][ty]
                        sum = sum // num if num > 0 else sum
                        new_data[i][j] = sum
        else:
            dir = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
            for i in range(new_height):
                for j in range(new_width):
                    if new_data[i][j][0] == 0 and new_data[i][j][1] == 0 and new_data[i][j][2] == 0:
                        blue, green, red = [], [], []
                        for x0, y0 in dir:
                            tx, ty = i + x0, j + y0
                            if 0 <= tx < new_height and 0 <= ty < new_width:
                                blue.append(new_data[tx][ty][0])
                                green.append(new_data[tx][ty][1])
                                red.append(new_data[tx][ty][2])
                        new_data[i][j][0] = np.average(blue)
                        new_data[i][j][1] = np.average(green)
                        new_data[i][j][2] = np.average(red)
        self.data = new_data
        self.biHeight = new_height
        self.biWidth = new_width
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 图像逆时针旋转，反变换，不用插值
    def Rotate_Counterclockwise(self, angle, save_path=default_save_path):
        angle = angle * math.pi / 180  # 角度弧度转化
        centerY, centerX = int(self.biWidth / 2), int(self.biHeight / 2)
        tmpx, tmpy = [], []
        # 旋转后的图像的像素的最大值和最小值仍然在四个顶点处取得
        for x0, y0 in ((0, 0), (0, self.biHeight - 1), (self.biWidth - 1, 0), (self.biHeight - 1, self.biWidth - 1)):
            x = (x0 - centerX) * math.cos(angle) - (y0 - centerY) * math.sin(angle) + centerX
            y = (x0 - centerX) * math.sin(angle) + (y0 - centerY) * math.cos(angle) + centerY
            tmpx.append(int(x + 0.5)), tmpy.append(int(y + 0.5))
        # 获得新图的长宽, 并声明新图区域
        new_height, new_width = int(max(tmpx) - min(tmpx)) + 1, int(max(tmpy) - min(tmpy)) + 1
        new_centerY, new_centerX = int(new_height / 2), int(new_width / 2)
        if not self.IsRealColor:
            new_data = np.zeros((new_height, new_width), np.uint8)
        else:
            new_data = np.zeros((new_height, new_width, self.data.shape[-1]), np.uint8)
        # 开始旋转 (注意这里的相对坐标关系)
        for i in range(new_height):
            for j in range(new_width):
                i0 = int((i - new_centerX) * math.cos(angle) + (j - new_centerY) * math.sin(angle) + centerX)
                j0 = int(-(i - new_centerX) * math.sin(angle) + (j - new_centerY) * math.cos(angle) + centerY)
                if 0 <= i0 < self.biHeight and 0 <= j0 < self.biWidth:
                    new_data[i][j] = self.data[i0][j0]
        self.data = new_data
        self.biHeight = new_height
        self.biWidth = new_width
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 图像顺时针旋转，反变换，不用插值
    def Rotate_Clockwise(self, angle, save_path=default_save_path):
        angle = angle * math.pi / 180  # 角度弧度转化
        centerY, centerX = int(self.biWidth / 2), int(self.biHeight / 2)
        tmpx, tmpy = [], []
        # 旋转后的图像的像素的最大值和最小值仍然在四个顶点处取得
        for x0, y0 in ((0, 0), (0, self.biHeight - 1), (self.biWidth - 1, 0), (self.biHeight - 1, self.biWidth - 1)):
            x = (x0 - centerX) * math.cos(angle) + (y0 - centerY) * math.sin(angle) + centerX
            y = -(x0 - centerX) * math.sin(angle) + (y0 - centerY) * math.cos(angle) + centerY
            tmpx.append(int(x + 0.5)), tmpy.append(int(y + 0.5))
        # 获得新图的长宽, 并声明新图区域
        new_height, new_width = int(max(tmpx) - min(tmpx)) + 1, int(max(tmpy) - min(tmpy)) + 1
        new_centerY, new_centerX = int(new_height / 2), int(new_width / 2)
        if not self.IsRealColor:
            new_data = np.zeros((new_height, new_width), np.uint8)
        else:
            new_data = np.zeros((new_height, new_width, self.data.shape[-1]), np.uint8)
        # 开始旋转 (注意这里的相对坐标关系)
        for i in range(new_height):
            for j in range(new_width):
                i0 = int((i - new_centerX) * math.cos(angle) - (j - new_centerY) * math.sin(angle) + centerX)
                j0 = int((i - new_centerX) * math.sin(angle) + (j - new_centerY) * math.cos(angle) + centerY)
                if 0 <= i0 < self.biHeight and 0 <= j0 < self.biWidth:
                    new_data[i][j] = self.data[i0][j0]
        self.data = new_data
        self.biHeight = new_height
        self.biWidth = new_width
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 图片的抽样缩小
    def Shrink(self, k1, k2, save_path=default_save_path):
        assert 0 < k1 <= 1 and 0 < k2 <= 1
        new_height = int(self.biHeight * k1 + 0.5)
        new_width = int(self.biWidth * k2 + 0.5)
        if not self.IsRealColor:
            new_data = np.zeros((new_height, new_width), np.uint8)
        else:
            new_data = np.zeros((new_height, new_width, self.data.shape[-1]), np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                new_data[i][j] = self.data[int(i / k1)][int(j / k2)]
        self.data = new_data
        self.biHeight = new_height
        self.biWidth = new_width
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 图片按同样比例缩小
    def Shrink_Same(self, k, save_path=default_save_path):
        self.Shrink(k, k, save_path)

    # 图像y轴错切(x轴不变), 这里的输入是角度制
    def Shear_Mapping_Y(self, angle, save_path=default_save_path):
        assert not (angle % 180 == 90 or angle < 0)
        ratio = math.tan(angle * math.pi / 180.0)
        new_height = int(ratio * self.biWidth + self.biHeight + 0.5)
        new_width = self.biWidth
        if not self.IsRealColor:
            new_data = np.zeros((new_height, new_width), np.uint8)
        else:
            new_data = np.zeros((new_height, new_width, self.data.shape[-1]), np.uint8)
        for i in range(self.biHeight):
            for j in range(self.biWidth):
                new_data[int(i + j * ratio + 0.5)][j] = self.data[i][j]
        self.data = new_data
        self.biHeight = new_height
        self.biWidth = new_width
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 图像x轴错切(y轴不变), 这里的输入是角度制
    def Shear_Mapping_X(self, angle, save_path=default_save_path):
        assert not (angle % 180 == 90 or angle < 0)
        ratio = math.tan(angle * math.pi / 180.0)
        new_width = int(ratio * self.biHeight + self.biWidth + 0.5)
        new_height = self.biHeight
        if not self.IsRealColor:
            new_data = np.zeros((new_height, new_width), np.uint8)
        else:
            new_data = np.zeros((new_height, new_width, self.data.shape[-1]), np.uint8)
        for i in range(self.biHeight):
            for j in range(self.biWidth):
                new_data[i][int(j + i * ratio + 0.5)] = self.data[i][j]
        self.data = new_data
        self.biHeight = new_height
        self.biWidth = new_width
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    # 图像按最近邻插值放大(基于像素的放大) k1是高度, k2是宽度
    def Nearest_Interpolation_Magnification(self, k1, k2, save_path=default_save_path):
        assert k1 >= 1 and k2 >= 1
        new_width = int(self.biWidth * k2 + 0.5)
        new_height = int(self.biHeight * k1 + 0.5)
        c1, c2 = 1 / k1, 1 / k2
        if not self.IsRealColor:
            new_data = np.zeros((new_height, new_width), np.uint8)
        else:
            new_data = np.zeros((new_height, new_width, self.data.shape[-1]), np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                new_i, new_j = int(i * c1), int(j * c2)
                new_data[i][j] = self.data[new_i][new_j]
        self.data = new_data
        self.biHeight = new_height
        self.biWidth = new_width
        self.SaveBMP(save_path)
        self.ShowPic(save_path)

    ''' 双线性插值放大算法
    https://zhuanlan.zhihu.com/p/110754637
    OpenCV中的双线性插值有两个优化： 中心的偏移问题 and 浮点数->整数的优化。 这里只用了第一个
    '''
    def Bilinear_Interpolation_Magnification(self, k1, k2, save_path=default_save_path):
        assert k1 >= 1 and k2 >= 1
        c1, c2 = 1 / k1, 1 / k2
        dstH, dstW = int(k1 * self.biHeight + 0.5), int(k2 * self.biWidth + 0.5)
        img, scrH, scrW = self.data, self.biHeight, self.biWidth
        if not self.IsRealColor:
            new_data = np.zeros((dstH, dstW), np.uint8)
            # img = np.pad(img, (1, 1), 'edge')
        else:
            new_data = np.zeros((dstH, dstW, self.data.shape[-1]), np.uint8)
            # img = np.pad(self.data, ((0, 1), (0, 1), (0, 0)), 'constant')
        for i in range(dstH):
            for j in range(dstW):
                scr_x, scr_y = (i + 0.5) * c1 - 0.5, (j + 0.5) * c2 - 0.5
                x, y = math.floor(scr_x), math.floor(scr_y)
                u, v = scr_x - x, scr_y - y
                xx = x + 1 if x + 1 < self.biHeight else self.biHeight - 1
                yy = y + 1 if y + 1 < self.biWidth else self.biWidth - 1
                new_data[i][j] = ((1 - u) * (1 - v) * img[x][y] + u * (1 - v) * img[xx][y]
                                  + (1 - u) * v * img[x][yy] + u * v * img[xx][yy])
        self.data = new_data
        self.biHeight = dstH
        self.biWidth = dstW
        self.SaveBMP(save_path)
        self.ShowPic(save_path)


# 真彩 lena512color24Bits.bmp   256色 lena512color_8Bits.bmp alpha lena512color32BitsAlpha.bmp
# 地毯 E:\大三上学期课内\数字图像\图像处理图片\rug.bmp


if __name__ == "__main__":
    path = r'E:\大三上学期课内\数字图像\图像处理图片\Lena\lena512color24Bits.bmp'
    a = BMP('./img/haze.bmp')
    # a.ShowPicInfo(True)
    a.Nearest_Interpolation_Magnification(2, 2)
    # a = np.arange(1, 13).reshape(3, 2, 2)
    # a = np.pad(a, ((0, 1), (0, 1), (0, 1)), 'constant')
    # print(a)
