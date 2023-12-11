# -*- coding : utf-8 -*-
# @Time : 2021-12-01 14:45
# @Author: gla1ve
# @File: STQ_UI.py
# @Software: PyCharm
# 人生苦短， 我用python(划掉) Java
import sys
import warnings

from PyQt5 import QtWidgets, QtGui

from BMP import BMP
from Strength import Strength
from Sharpen import Sharpen
from Segmentation import Segmentation
from STQ数字图像处理 import Ui_mainWindow
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QInputDialog, QLineEdit
from qt_material import apply_stylesheet


class STQForm(QtWidgets.QMainWindow, Ui_mainWindow):
    def __init__(self):
        self.default_save_path = 'img/default_save.bmp'
        super(STQForm, self).__init__()
        self.setupUi(self)
        QMessageBox.about(self.window(), "使用提示", "欢迎使用~ 在对图像进行操作前，请先选择要处理的图片~")
        self.pic_path = ""  # 初始化空值
        self.menubar.setEnabled(False)  # 初始化的时候没有打开图片，菜单不可用

    # 打开图片
    def OpenImg(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.bmp;")
        jpg = QtGui.QPixmap(imgName).scaled(self.ori_pic.width(), self.ori_pic.height())
        self.ori_pic.setPixmap(jpg)
        self.origin_pic_path.setText(imgName)
        if str(imgName).startswith("E:/数字图像处理STQ/"):
            imgName = str(imgName).replace("E:/数字图像处理STQ/", "")
        self.pic_path = imgName
        self.fixed_pic.setText("")
        if self.pic_path == "":
            self.menubar.setEnabled(False)
            return
        self.menubar.setEnabled(True)

    # 显示保存的图片
    def Show_Fixed_Pic(self, path):
        jpg = QtGui.QPixmap(path).scaled(self.fixed_pic.width(), self.fixed_pic.height())
        self.fixed_pic.setPixmap(jpg)
        self.save_pic_path.setText(path)

    # 显示位图信息头
    def Show_Pic_Info(self):
        pass

    '''
    以下为锐化部分槽函数
    '''

    # 一阶线性水平锐化槽函数
    def Horizontal_Sharpen_slot(self):
        sharpen = Sharpen(self.pic_path)
        sharpen.Horizontal_Sharpen()
        self.Show_Fixed_Pic('img/Sharpen.bmp')

    # 一阶线性垂直锐化槽函数
    def Vertical_Sharpen_slot(self):
        sharpen = Sharpen(self.pic_path)
        sharpen.Vertical_Sharpen()
        self.Show_Fixed_Pic('img/Sharpen.bmp')

    # 基于门限值的梯度锐化槽函数
    def Gradient_Sharpen_slot(self):
        sharpen = Sharpen(self.pic_path)
        sharpen.Gradient_Sharpen()
        self.Show_Fixed_Pic('img/Sharpen.bmp')

    # Robert锐化槽函数
    def Robert_Sharpen_slot(self):
        sharpen = Sharpen(self.pic_path)
        sharpen.Robert_Sharpen()
        self.Show_Fixed_Pic('img/Sharpen.bmp')

    # Sobel锐化槽函数
    def Sobel_Sharpen_slot(self):
        sharpen = Sharpen(self.pic_path)
        sharpen.Sobel_Sharpen()
        self.Show_Fixed_Pic('img/Sharpen.bmp')

    # Prewitt锐化槽函数
    def Prewitt_Sharpen_slot(self):
        sharpen = Sharpen(self.pic_path)
        sharpen.Prewitt_Sharpen()
        self.Show_Fixed_Pic('img/Sharpen.bmp')

    # Wallis锐化槽函数
    def Wallis_Sharpen_slot(self):
        sharpen = Sharpen(self.pic_path)
        sharpen.Wallis_Sharpen()
        self.Show_Fixed_Pic('img/Sharpen.bmp')

    # Laplacian锐化槽函数
    def Laplacian_Sharpen_slot(self):
        sharpen = Sharpen(self.pic_path)
        sharpen.Laplacian_Sharpen()
        self.Show_Fixed_Pic('img/Sharpen.bmp')

    # Laplacian2锐化槽函数
    def Laplacian2_Sharpen_slot(self):
        sharpen = Sharpen(self.pic_path)
        sharpen.Laplacian2_Sharpen()
        self.Show_Fixed_Pic('img/Sharpen.bmp')

    '''
    以下为去噪相关部分槽函数
    '''
    # 显示图像直方图槽函数
    def ShowHist_slot(self):
        strength = Strength(self.pic_path)
        strength.ShowHist()
        self.Show_Fixed_Pic(self.pic_path)

    # 直方图均衡化槽函数
    def Histogram_Equalization2_slot(self):
        strength = Strength(self.pic_path)
        strength.Histogram_Equalization2()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # 直方图均衡化槽函数[分开的]
    def Histogram_Equalization_slot(self):
        strength = Strength(self.pic_path)
        strength.Histogram_Equalization()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # 二值化槽函数
    def Binarization_slot(self):
        strength = Strength(self.pic_path)
        strength.Binarization()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # 灰窗级切片槽函数
    def Grey_Cutter_slot(self):
        strength = Strength(self.pic_path)
        strength.Grey_Cutter()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # 灰度化槽函数
    def Grey_Pic_slot(self):
        strength = Strength(self.pic_path)
        strength.Grey_Pic()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # 添加随机噪声槽函数
    def Generate_Random_Noise_slot(self):
        strength = Strength(self.pic_path)
        strength.Generate_Random_Noise()
        self.Show_Fixed_Pic('img/Noised.bmp')

    # 添加椒盐噪声槽函数
    def Generate_Salt_And_Pepper_Noise_slot(self):
        strength = Strength(self.pic_path)
        strength.Generate_Salt_And_Pepper_Noise()
        self.Show_Fixed_Pic('img/Noised.bmp')

    # 添加Gauss噪声槽函数
    def Generate_Gauss_Noise_slot(self):
        strength = Strength(self.pic_path)
        strength.Generate_Gauss_Noise()
        self.Show_Fixed_Pic('img/Noised.bmp')

    # 黑白点噪声滤波槽函数
    def Black_And_White_Dot_Noise_Filtering_slot(self):
        strength = Strength(self.pic_path)
        strength.Black_And_White_Dot_Noise_Filtering()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # Gauss滤波槽函数
    def Gaussian_Blur_slot(self):
        strength = Strength(self.pic_path)
        strength.Gaussian_Blur()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # 均值滤波槽函数
    def Average_Filtering_slot(self):
        strength = Strength(self.pic_path)
        strength.Average_Filtering()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # 中值滤波槽函数
    def Median_Filtering_slot(self):
        strength = Strength(self.pic_path)
        strength.Median_Filtering()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # SNN滤波槽函数
    def SNN_Filtering_slot(self):
        QMessageBox.about(self.window(), "SNN提示", "SNN滤波时间较慢... 请耐心等待yo~")
        strength = Strength(self.pic_path)
        strength.SNN_Filtering()
        self.Show_Fixed_Pic('img/Strength.bmp')

    # KNN滤波槽函数
    def KNN_Filtering_slot(self):
        QMessageBox.about(self.window(), "KNN提示", "KNN滤波时间较慢... 请耐心等待yo~")
        strength = Strength(self.pic_path)
        strength.KNN_Filtering()
        self.Show_Fixed_Pic('img/Strength.bmp')

    '''
    以下为图像几何部分相关部分槽函数
    '''
    # 显示图片信息
    def Show_Info_slot(self):
        bmp = BMP(self.pic_path)
        ss = bmp.ShowPicInfo()
        self.fixed_pic.setText(ss)

    # 不改变大小的平移槽函数。没写异常处理
    def Move_slot(self):
        bmp = BMP(self.pic_path)
        value, ok = QInputDialog.getText(self, "请输入平移参数~", "输入格式为a,b(没有空格).\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "20,20")
        try:
            ipt = value.split(",")
            a, b = [float(x) for x in ipt]
            bmp.Move(a, b)
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # 改变大小的平移槽函数。没写异常处理
    def Move_Without_Changing_Size_slot(self):
        bmp = BMP(self.pic_path)
        value, ok = QInputDialog.getText(self, "请输入平移参数~", "输入格式为a,b(没有空格).\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "20,20")
        try:
            ipt = value.split(",")
            a, b = [float(x) for x in ipt]
            bmp.Move_Without_Changing_Size(a, b)
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # 水平镜像槽函数
    def Mirror_Horizontally_slot(self):
        bmp = BMP(self.pic_path)
        bmp.Mirror_Horizontally()
        self.Show_Fixed_Pic('img/bmp_save.bmp')

    # 垂直镜像槽函数
    def Mirror_Vertically_slot(self):
        bmp = BMP(self.pic_path)
        bmp.Mirror_Vertically()
        self.Show_Fixed_Pic('img/bmp_save.bmp')

    # 中心对称槽函数
    def Central_symmetry_slot(self):
        bmp = BMP(self.pic_path)
        bmp.Central_symmetry()
        self.Show_Fixed_Pic('img/bmp_save.bmp')

    # 逆时针旋转槽函数
    def Rotate_Counterclockwise_slot(self):
        value, ok = QInputDialog.getText(self, "请输入旋转角度~", "输入格式为一个实数.\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "30")
        try:
            value = float(value)
            bmp = BMP(self.pic_path)
            bmp.Rotate_Counterclockwise(angle=value)
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # 顺时针旋转槽函数
    def Rotate_Clockwise_slot(self):
        value, ok = QInputDialog.getText(self, "请输入旋转角度~", "输入格式为一个实数.\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "30")
        try:
            value = float(value)
            bmp = BMP(self.pic_path)
            bmp.Rotate_Clockwise(angle=value)
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # 图片的抽样缩小槽函数
    def Shrink_slot(self):
        value, ok = QInputDialog.getText(self, "请输入缩小参数[x, y]~", "输入格式为a,b(没有空格,0-1实数).\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "0.5,0.5")
        try:
            ipt = value.split(",")
            a, b = [float(x) for x in ipt]
            bmp = BMP(self.pic_path)
            bmp.Shrink(k1=a, k2=b)
            QMessageBox.about(self.window(), "缩小提示", "这里处理完图片的大小是固定的,不易看出缩小的结果。请去本地保存查看缩小后的图片~")
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # 图像同样比例缩小槽函数
    def Shrink_Same_slot(self):
        value, ok = QInputDialog.getText(self, "请输入缩小参数~", "输入格式为0-1实数.\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "0.5")
        try:
            value = float(value)
            bmp = BMP(self.pic_path)
            bmp.Shrink_Same(k=value)
            QMessageBox.about(self.window(), "缩小提示", "这里处理完图片的大小是固定的,不易看出缩小的结果。请去本地保存查看缩小后的图片~")
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # Y轴错切槽函数
    def Shear_Mapping_Y_slot(self):
        value, ok = QInputDialog.getText(self, "请输入错切参数~", "输入为正角度制[no tan90].\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "45")
        try:
            value = float(value)
            bmp = BMP(self.pic_path)
            bmp.Shear_Mapping_Y(angle=value)
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # X轴错切槽函数
    def Shear_Mapping_X_slot(self):
        value, ok = QInputDialog.getText(self, "请输入错切参数~", "输入为正角度制[no tan90].\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "45")
        try:
            value = float(value)
            bmp = BMP(self.pic_path)
            bmp.Shear_Mapping_X(angle=value)
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # 双线性插值放大槽函数
    def Nearest_Interpolation_Magnification_slot(self):
        value, ok = QInputDialog.getText(self, "请输入缩小参数[高, 宽]~", "输入格式为a,b(没有空格,>1实数).\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "2,2")
        try:
            ipt = value.split(",")
            a, b = [float(x) for x in ipt]
            bmp = BMP(self.pic_path)
            bmp.Nearest_Interpolation_Magnification(k1=a, k2=b)
            QMessageBox.about(self.window(), "放大提示", "这里处理完图片的大小是固定的,不易看出放大的结果。请去本地保存查看放大后的图片~")
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # 双线性插值放大槽函数
    def Bilinear_Interpolation_Magnification_slot(self):
        value, ok = QInputDialog.getText(self, "请输入缩小参数[高, 宽]~", "输入格式为a,b(没有空格,>1实数).\nP.S.我没加异常处理，乱输入程序会崩哦",
                                         QLineEdit.Normal, "2,2")
        try:
            ipt = value.split(",")
            a, b = [float(x) for x in ipt]
            bmp = BMP(self.pic_path)
            bmp.Bilinear_Interpolation_Magnification(k1=a, k2=b)
            QMessageBox.about(self.window(), "放大提示", "这里处理完图片的大小是固定的,不易看出放大的结果。请去本地保存查看放大后的图片~")
            self.Show_Fixed_Pic('img/bmp_save.bmp')
        except Exception as e:
            QMessageBox.about(self.window(), "提示", f"{e}\n输入有误! 请不要玩火emm")

    # 以下分割部分
    # 基于灰度直方图的阈值分割槽函数
    def Gray_Histogram_Segmentation_slot(self):
        segmentation = Segmentation(self.pic_path)
        segmentation.Gray_Histogram_Segmentation()
        self.Show_Fixed_Pic('img/Segmentation.bmp')

    # 最小类内方差法槽函数
    def Minimum_Variance_slot(self):
        segmentation = Segmentation(self.pic_path)
        segmentation.Minimum_Variance()
        self.Show_Fixed_Pic('img/Segmentation.bmp')

    # P参数生长槽函数
    def P_Parameter_slot(self):
        segmentation = Segmentation(self.pic_path)
        segmentation.P_Parameter()
        self.Show_Fixed_Pic('img/Segmentation.bmp')

    # 腐蚀槽函数
    def Erosion_slot(self):
        segmentation = Segmentation(self.pic_path)
        segmentation.Erosion()
        self.Show_Fixed_Pic('img/Segmentation.bmp')

    # 膨胀槽函数
    def Inflation_slot(self):
        segmentation = Segmentation(self.pic_path)
        segmentation.Inflation()
        self.Show_Fixed_Pic('img/Segmentation.bmp')

    # 闭运算槽函数
    def Close_Operation_slot(self):
        segmentation = Segmentation(self.pic_path)
        segmentation.Close_Operation()
        self.Show_Fixed_Pic('img/Segmentation.bmp')

    # 开运算槽函数
    def Open_Operation_slot(self):
        segmentation = Segmentation(self.pic_path)
        segmentation.Open_Operation()
        self.Show_Fixed_Pic('img/Segmentation.bmp')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    app = QtWidgets.QApplication(sys.argv)
    now_Status = STQForm()
    apply_stylesheet(app, theme='dark_purple.xml')
    now_Status.show()
    sys.exit(app.exec_())