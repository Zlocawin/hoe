import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


# brenner梯度函数计算
def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out += (int(img[x+2, y])-int(img[x, y]))**2
    return out


def Tenengrad(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    G = np.sqrt(sobelx ** 2 + sobely ** 2)
    out = np.sum(G)
    return out


# Laplacian梯度函数计算
def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    return cv2.Laplacian(img, cv2.CV_64F).var()


# SMD梯度函数计算
def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0]-1):
        for y in range(0, shape[1]):
            out+=math.fabs(int(img[x, y])-int(img[x, y-1]))
            out+=math.fabs(int(img[x, y]-int(img[x+1, y])))
    return out


# SMD2梯度函数计算
def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x, y])-int(img[x+1, y]))*math.fabs(int(img[x, y]-int(img[x, y+1])))
    return out


# 方差函数计算
def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            out+=(img[x, y]-u)**2
    return out


# energy函数计算
def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1, y])-int(img[x, y]))**2)*((int(img[x, y+1]-int(img[x, y])))**2)
    return out


#  Vollath函数计算
def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x, y])*int(img[x+1, y])
    return out


#  entropy函数计算
def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out


def norm(score1, score2):
    max = np.max([score1, score2])
    return score1/max, score2/max


def main(img1, img2):
    print('Brenner', norm(brenner(img1), brenner(img2)))
    print('Tenengrad', norm(Tenengrad(img1), Tenengrad(img2)))
    print('Laplacian', norm(Laplacian(img1), Laplacian(img2)))
    print('SMD', norm(SMD(img1), SMD(img2)))
    print('SMD2', norm(SMD2(img1), SMD2(img2)))
    print('Variance', norm(variance(img1), variance(img2)))
    print('Energy', norm(energy(img1), energy(img2)))
    print('Vollath', norm(Vollath(img1), Vollath(img2)))
    print('Entropy', norm(entropy(img1), entropy(img2)))


if __name__ == '__main__':
    #  读入原始图像
    img1 = cv2.imread('.\\GOPRO\\GOPRO_3840FPS_AVG_3-21\\test\\blur\\GOPR0862_11_00\\003953_17.png')
    img2 = cv2.imread('.\\GOPRO\\GOPRO_3840FPS_AVG_3-21\\test\\sharp\\GOPR0862_11_00\\003953_17.png')

    plt.figure()
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    #  灰度化处理
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    main(img1, img2)
