from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
import numpy as np
import math
import time
import random
from scipy.ndimage import convolve
from skimage.filters.edges import HSOBEL_WEIGHTS, VSOBEL_WEIGHTS
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import matplotlib.ticker as ticker
import os
import csv


matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.rcParams['axes.unicode_minus'] = False


def marziliano_method_h(edges, image):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    Calculate the widths of the given edges.

    :return: A matrix with the same dimensions as the given image with 0's at
        non-edge locations and edge-widths at the edge locations.
    """

    # `edge_widths` consists of zero and non-zero values. A zero value
    # indicates that there is no edge at that position and a non-zero value
    # indicates that there is an edge at that position and the value itself
    # gives the edge width.
    edge_widths = np.zeros(image.shape)

    # find the gradient for the image
    gradient_y, gradient_x = np.gradient(image)

    # dimensions of the image
    img_height, img_width = image.shape

    # holds the angle information of the edges
    edge_angles = np.zeros(image.shape)

    edge_ignore = 0.1    # 不参与边缘展宽计算的边界像素的比例

    finetune_number = 0  # 微调判断的灰度值

    # calculate the angle of the edges
    for row in range(img_height):
        for col in range(img_width):
            if gradient_x[row, col] != 0:
                edge_angles[row, col] = math.atan2(gradient_y[row, col], gradient_x[row, col]) * (180 / np.pi)
            elif gradient_x[row, col] == 0 and gradient_y[row, col] == 0:
                edge_angles[row, col] = 0
            elif gradient_x[row, col] == 0 and gradient_y[row, col] == np.pi/2:
                edge_angles[row, col] = 90


    if np.any(edge_angles):

        # quantize the angle
        # quantized_angles = 45 * np.round(edge_angles / 45)
        quantized_angles = 180 * np.round(edge_angles / 180)

        for row in range(1, img_height - 1):
            for col in range(int(edge_ignore * img_width), int((1 - edge_ignore) * (img_width - 1))):
                if edges[row, col] == 1:

                    # gradient angle = 180 or -180
                    if quantized_angles[row, col] == 180 or quantized_angles[row, col] == -180:
                        for margin in range(100 + 1):
                            inner_border = (col - 1) - margin
                            outer_border = (col - 2) - margin

                            # outside image or intensity increasing from left to right
                            if outer_border < 0 or (image[row, outer_border] - image[row, inner_border]) <= -1:
                                outer_border += 1
                                for finetune in range(margin+1):
                                    inner_border += 1
                                    # outer_border += 1
                                    if (image[row, outer_border] - image[row, inner_border]) > finetune_number:
                                        break
                                margin = margin - finetune
                                break

                        width_left = margin + 1

                        for margin in range(100 + 1):
                            inner_border = (col + 1) + margin
                            outer_border = (col + 2) + margin

                            # outside image or intensity increasing from left to right
                            if outer_border >= img_width or (image[row, outer_border] - image[row, inner_border]) >= 1:
                                outer_border -= 1
                                for finetune in range(margin+1):
                                    inner_border -= 1
                                    # outer_border -= 1
                                    if (image[row, inner_border] - image[row, outer_border]) > finetune_number:
                                        break
                                margin = margin - finetune
                                break

                        width_right = margin + 1

                        edge_widths[row, col] = width_left + width_right


                    # gradient angle = 0
                    if quantized_angles[row, col] == 0:
                        for margin in range(100 + 1):
                            inner_border = (col - 1) - margin
                            outer_border = (col - 2) - margin

                            # outside image or intensity decreasing from left to right
                            if outer_border < 0 or (image[row, outer_border] - image[row, inner_border]) >= 1:
                                outer_border += 1
                                for finetune in range(margin+1):
                                    inner_border += 1
                                    # outer_border += 1
                                    if (image[row, inner_border] - image[row, outer_border]) > finetune_number:
                                        break
                                margin = margin - finetune
                                break

                        width_left = margin + 1

                        for margin in range(100 + 1):
                            inner_border = (col + 1) + margin
                            outer_border = (col + 2) + margin

                            # outside image or intensity decreasing from left to right
                            if outer_border >= img_width or (image[row, outer_border] - image[row, inner_border]) <= -1:
                                outer_border -= 1
                                for finetune in range(margin+1):
                                    inner_border -= 1
                                    # outer_border -= 1
                                    if (image[row, outer_border] - image[row, inner_border]) > finetune_number:
                                        break
                                margin = margin - finetune
                                break

                        width_right = margin + 1

                        edge_widths[row, col] = width_right + width_left

    return edge_widths


def marziliano_method_v(edges, image):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    Calculate the widths of the given edges.

    :return: A matrix with the same dimensions as the given image with 0's at
        non-edge locations and edge-widths at the edge locations.
    """

    # `edge_widths` consists of zero and non-zero values. A zero value
    # indicates that there is no edge at that position and a non-zero value
    # indicates that there is an edge at that position and the value itself
    # gives the edge width.
    edge_widths = np.zeros(image.shape)

    # find the gradient for the image
    gradient_y, gradient_x = np.gradient(image)

    # dimensions of the image
    img_height, img_width = image.shape

    # holds the angle information of the edges
    edge_angles = np.zeros(image.shape)

    edge_ignore = 0.1       # 不参与边缘展宽计算的边界像素的比例

    finetune_number = 0     # 微调判断的灰度值

    # calculate the angle of the edges
    for row in range(img_height):
        for col in range(img_width):
            if gradient_x[row, col] != 0:
                edge_angles[row, col] = math.atan2(gradient_y[row, col], gradient_x[row, col]) * (180 / np.pi)
            elif gradient_x[row, col] == 0 and gradient_y[row, col] == 0:
                edge_angles[row, col] = 0
            elif gradient_x[row, col] == 0 and gradient_y[row, col] == np.pi/2:
                edge_angles[row, col] = 90


    if np.any(edge_angles):

        # quantize the angle
        # quantized_angles = 45 * np.round(edge_angles / 45)
        quantized_angles = 180 * np.round((edge_angles - 90) / 180) + 90

        for row in range(int(edge_ignore * img_height), int((1 - edge_ignore) * (img_height - 1))):
            for col in range(1, img_width - 1):
                if edges[row, col] == 1:

                    # gradient angle = -90
                    if quantized_angles[row, col] == -90:
                        for margin in range(100 + 1):
                            inner_border = (row - 1) - margin
                            outer_border = (row - 2) - margin

                            # outside image or intensity decreasing from up to down
                            if outer_border < 0 or (image[outer_border, col] - image[inner_border, col]) <= -1:
                                outer_border += 1
                                for finetune in range(margin+1):
                                    inner_border += 1
                                    if (image[outer_border, col] - image[inner_border, col]) > finetune_number:
                                        break
                                margin = margin - finetune
                                break

                        width_up = margin + 1

                        for margin in range(100 + 1):
                            inner_border = (row + 1) + margin
                            outer_border = (row + 2) + margin

                            # outside image or intensity decreasing from up to down
                            if outer_border >= img_height or (image[outer_border, col] - image[inner_border, col]) >= 1:
                                outer_border -= 1
                                for finetune in range(margin+1):
                                    inner_border -= 1
                                    if (image[inner_border, col] - image[outer_border, col]) > finetune_number:
                                        break
                                margin = margin - finetune
                                break

                        width_down = margin + 1

                        edge_widths[row, col] = width_up + width_down

                    # gradient angle = 90
                    if quantized_angles[row, col] == 90:
                        for margin in range(100 + 1):
                            inner_border = (row - 1) - margin
                            outer_border = (row - 2) - margin

                            # outside image or intensity increasing from up to down
                            if outer_border < 0 or (image[outer_border, col] - image[inner_border, col]) >= 1:
                                outer_border += 1
                                for finetune in range(margin+1):
                                    inner_border += 1
                                    if (image[inner_border, col] - image[outer_border, col]) > finetune_number:
                                        break
                                margin = margin - finetune
                                break

                        width_up = margin + 1

                        for margin in range(100 + 1):
                            inner_border = (row + 1) + margin
                            outer_border = (row + 2) + margin

                            # outside image or intensity decreasing from up to down
                            if outer_border >= img_height or (
                                    image[outer_border, col] - image[inner_border, col]) <= -1:
                                outer_border -= 1
                                for finetune in range(margin+1):
                                    inner_border -= 1
                                    if (image[outer_border, col] - image[inner_border, col]) > finetune_number:
                                        break
                                margin = margin - finetune
                                break

                        width_down = margin + 1

                        edge_widths[row, col] = width_up + width_down

    return edge_widths


def sobel_h(image):
    # type: (numpy.ndarray) -> numpy.ndarray
    """
    Find edges using the Sobel approximation to the derivatives.

    Inspired by the [Octave implementation](https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l196).
    """

    h1 = np.array(HSOBEL_WEIGHTS)       # 3*3size
    # 5*5 size
    # h1 = np.array([[1, 2, 0, -2, -1], [4, 8, 0, -8, -4], [6, 12, 0, -12, -6], [4, 8, 0, -8, -4], [1, 2, 0, -2, -1]], dtype=float).T
    h1 /= np.sum(abs(h1))  # normalize h1

    strength2 = np.square(convolve(image, h1.T))

    # Note: https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l59
    # thresh2 = 30 * np.sqrt(np.mean(strength2))

    # 自适应阈值
    strength1 = strength2.flatten()     # 将二维数组展为一维
    strength = np.sort(strength1)       # 排序
    index = int(strength.size * 0.99)   # 需要筛除掉的像素比例
    thresh2 = strength[index]           # 自适应阈值

    strength2[strength2 <= thresh2] = 0
    return _simple_thinning(strength2)


def sobel_v(image):
    # type: (numpy.ndarray) -> numpy.ndarray
    """
    Find edges using the Sobel approximation to the derivatives.

    Inspired by the [Octave implementation](https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l196).
    """

    h1 = np.array(VSOBEL_WEIGHTS)       # 3*3
    # 5*5
    # h1 = np.array([[1, 2, 0, -2, -1], [4, 8, 0, -8, -4], [6, 12, 0, -12, -6], [4, 8, 0, -8, -4], [1, 2, 0, -2, -1]], dtype=float)
    h1 /= np.sum(abs(h1))  # normalize h1

    strength2 = np.square(convolve(image, h1.T))
    # Image.fromarray(np.uint8(strength2 * 255)).show()

    # Note: https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l59
    # thresh2 = 30 * np.sqrt(np.mean(strength2))

    # 自适应阈值
    strength1 = strength2.flatten()      # 将二维数组展为一维
    strength = np.sort(strength1)        # 排序
    index = int(strength.size * 0.99)    # 需要筛除掉的像素比例
    thresh2 = strength[index]            # 自适应阈值

    strength2[strength2 <= thresh2] = 0
    # Image.fromarray(np.uint8(strength2 * 255)).show()
    return _simple_thinning(strength2)


def _simple_thinning(strength):
    # type: (numpy.ndarray) -> numpy.ndarray
    """
    Perform a very simple thinning.

    Inspired by the [Octave implementation](https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l512).
    """
    num_rows, num_cols = strength.shape

    zero_column = np.zeros((num_rows, 1))
    zero_row = np.zeros((1, num_cols))

    # 若一个点比左右的sobel值都大，则为1，否则为0
    x = (
            (strength > np.c_[zero_column, strength[:, :-1]]) &
            (strength > np.c_[strength[:, 1:], zero_column])
    )

    # 若一个点比上下的sobel值都大，则为1，否则为0
    y = (
            (strength > np.r_[zero_row, strength[:-1, :]]) &
            (strength > np.r_[strength[1:, :], zero_row])
    )

    return x | y


# 判断块中边缘像素比例
def get_edgepixel_ratio(img_block, edges):
    m, n = img_block.shape
    L = m * n
    img_edge_h_pixels = np.sum(edges)
    ratio = img_edge_h_pixels / L
    return ratio


# 全局灰度线性变换，用于边缘展宽可视化
def global_linear_transmation(img1):  # 将灰度范围设为0~255
    img = img1.copy()
    maxV = img.max()
    minV = img.min()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0:
                img[i, j] = ((img[i, j]-minV)*255)/(maxV-minV)
    return img


# 绘制边缘曲线
def draw_edge(ori_edges, ori_img, img_filted, num):
    edges = ori_edges.copy()
    display_range = 70      # 边缘点左右边缘曲线显示的范围
    height, width = edges.shape
    for i in range(0, height):
        for j in range(0, 201):
            edges[i][j] = 0
    for i in range(height):
        for j in range(width - 201, width):
            edges[i][j] = 0
    grayscale_list = []     # 选出边缘的灰度值
    grayscale_filted = []
    for i in range(num):
        grayscale_list.append([])
        grayscale_filted.append([])
    edges_nonzero_index = np.nonzero(edges)
    nonzero_num = len(edges_nonzero_index[0])
    random_edges = random.sample(range(0, nonzero_num), num)
    for edge_i in range(num):
        for col_now in range(edges_nonzero_index[1][random_edges[edge_i]] - display_range, edges_nonzero_index[1][random_edges[edge_i]] + display_range):
            grayscale_list[edge_i].append(ori_img[edges_nonzero_index[0][random_edges[edge_i]]][col_now])
            grayscale_filted[edge_i].append(img_filted[edges_nonzero_index[0][random_edges[edge_i]]][col_now])

    x = list(range(-display_range, display_range, 1))
    plt.ion()
    plt.figure(num='边缘曲线')

    for i in range(1, num+1):
        plt.subplot(2, int(num / 2), i)
        plt.plot(x, grayscale_list[i-1], color='c')

    plt.figure(num='滤波后边缘曲线')

    for i in range(1, num + 1):
        plt.subplot(2, int(num / 2), i)
        plt.plot(x, grayscale_filted[i - 1], color='c')

    plt.ioff()
    plt.show()
    return 0


# 计算距离
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


# 最远点采样
def FPS(sample, num):
    '''sample:初始点数据(array类型),
    num:需要采样的数据点个数'''
    n = sample.shape[0]
    center = np.mean(sample, axis=0)  # 点云重心
    selected = []  # 储存采集点索引
    min_distance = []
    for i in range(n):
        min_distance.append(distance(sample[i], center))
    p0 = np.argmax(min_distance)
    selected.append(p0)  # 选距离重心最远点p0
    min_distance = []
    for i in range(n):
        min_distance.append(distance(sample[p0], sample[i]))
    selected.append(np.argmax(min_distance))
    for i in range(num - 2):
        for p in range(n):
            d = distance(sample[selected[-1]], sample[p])
            if d <= min_distance[p]:
                min_distance[p] = d
        selected.append(np.argmax(min_distance))
    return selected, sample[selected]


# 计算输入图像x、y模糊度和模值模糊度
# 输入：待评估灰度图像
# 输出：x方向、y方向及模值模糊度
def compute_motion_blur(img_name, image_path, rotation_angle):
    image_color = Image.open(image_path)
    # image_color.save('E:\\传统NR-IQA方法复现\\实拍图片\\华为4.17\\handheld_less_texture(320)\\' + img_name + '.jpg')
    image = image_color.convert('L')

    # 若需要旋转
    if rotation_angle != 0:
        image_color = image_color.rotate(rotation_angle, expand=1)

        # 创建一个全黑图像并在边界处生成白边用去去除原图旋转后由于旋转导致的边缘点
        image1 = Image.new('RGB', image.size, color='black')

        # 获取图像的宽度和高度
        width, height = image1.size

        # 在图像周围添加一条宽度为3像素的白边
        for x in range(width):
            for y in range(height):
                if x < 3 or x > width - 4 or y < 3 or y > height - 4:
                    image1.putpixel((x, y), (255, 255, 255))

        # 旋转图像
        image1_rotated = image1.rotate(rotation_angle, expand=1)
        image1_rotated = image1_rotated.convert('L')
        img1_rotated = np.array(image1_rotated, dtype=np.float32)

        image = image.rotate(rotation_angle, expand=1)

    img = np.array(image, dtype=np.float32)

    height, width = img.shape
    b_w = 320                # 图像块的宽度
    b_h = b_w                # 图像块的高度
    L = b_w * b_h           # 图像块的尺寸
    ratiolisth = []         # 记录块的边缘像素比例的列表
    ratiolistv = []         # 记录块的垂直方向边缘像素的列表
    ignore_ratio = 0.1      # 各块不参与排序的边缘像素的比例

    indexlist_h = []        # 记录最初选块的index
    selected_index_h = []     # 记录选中块的index
    coordinate_list_h = []         # 记录最初选块的坐标
    indexlist_v = []
    selected_index_v = []
    coordinate_list_v = []
    block_num_h_ori = 20      # 水平方向最初选块的数目
    block_num_v_ori = block_num_h_ori   # 垂直方向最初选块的数目
    blocknum_h = 9          # 水平方向需要的块的数目
    blocknum_v = blocknum_h     # 垂直方向需要的块数

    blocknum_final = 5      # 用来计算最终模糊度的块的个数

    bbox_list_h = []          # 画选出的水平框
    bbox_list_v = []          # 选出的垂直框
    bbox_list = []            # 调试用
    bbox_num = 0            # 框的个数

    bbox_num1 = bbox_num            # 水平框的个数
    bbox_num2 = bbox_num            # 垂直框的个数

    blockrow = int(np.floor(height / b_h))        # 图像块行
    blockcol = int(np.floor(width / b_w))        # 图像块列数
    sobel_h_edges = sobel_h(img)
    sobel_v_edges = sobel_v(img)
    # Image.fromarray(np.uint8(sobel_h_edges * 255)).save(
    #     'E:\\传统NR-IQA方法复现\\实拍图片\\左右分割图像\\'+img_name+'_x方向边缘(0.5%).jpg')
    #
    # Image.fromarray(np.uint8(sobel_v_edges * 255)).save(
    #     'E:\\传统NR-IQA方法复现\\实拍图片\\左右分割图像\\'+img_name+'_y方向边缘(0.5%).jpg')

    img_hfilted = img.copy()
    img_vfilted = img.copy()

    b, a = signal.butter(8, 0.1, 'lowpass')
    for i in range(height):
        img_hfilted[i] = signal.filtfilt(b, a, img_hfilted[i]).astype(int)      # 对每行进行低通滤波

    for i in range(width):
        img_vfilted[:, i] = signal.filtfilt(b, a, img_vfilted[:, i]).astype(int)    # 对每列进行低通滤波

    img_hfilted = np.clip(img_hfilted, 0, 255)
    img_vfilted = np.clip(img_vfilted, 0, 255)  # 范围截断在0-255范围

    # sobel_h_edges = sobel_h(img_hfilted)
    # sobel_v_edges = sobel_v(img_vfilted)

    # Image.fromarray(np.uint8(sobel_h_edges * 255)).save(
    #     'E:\\传统NR-IQA方法复现\\实拍图片\\华为4.17\\handheld-4\\AutoCapture_20230414_120138\\'+img_name+'_edge-pixel.jpg')

    # 去除由于旋转的黑边导致的边缘点
    if rotation_angle != 0:
        sobel_h_edges[img1_rotated > 0] = False
        sobel_h_edges[img == 0] = False
        sobel_v_edges[img1_rotated > 0] = False
        sobel_v_edges[img == 0] = False

    edgewidth_h = np.zeros((blockrow, blockcol))        # 记录各块水平边缘宽度
    edgewidth_v = np.zeros((blockrow, blockcol))        # 记录各块垂直边缘宽度

    edgewidth_h_std = np.zeros((blockrow, blockcol))    # 记录各块水平边缘宽度的标准差
    edgewidth_v_std = np.zeros((blockrow, blockcol))    # 记录各块垂直边缘宽度的标准差

    for i in range(1, blockrow + 1):
        for j in range(1, blockcol + 1):
            edges_h_temp = sobel_h_edges[(b_h * (i - 1)):(b_h * i), (b_w * (j - 1)):(b_w * j)]
            edges_h_temp[:, :int(ignore_ratio * b_w)] = 0            # 边界像素置0，不参与排序
            edges_h_temp[:, int((1 - ignore_ratio) * b_w):] = 0
            edges_v_temp = sobel_v_edges[(b_h * (i - 1)):(b_h * i), (b_w * (j - 1)):(b_w * j)]
            edges_v_temp[:int(ignore_ratio * b_h), :] = 0
            edges_v_temp[int((1 - ignore_ratio) * b_h):, :] = 0
            ratio_temp_h = np.sum(edges_h_temp) / L        # 水平方向边缘像素比例
            ratio_temp_v = np.sum(edges_v_temp) / L        # 垂直方向边缘像素比例
            ratiolisth.append(ratio_temp_h)
            ratiolistv.append(ratio_temp_v)

    ratio_index = np.argsort(ratiolisth)  # 块的水平边缘像素比例从小到大排序的索引
    indexflag = 1           # 索引

    while block_num_h_ori:
        indexnow = ratio_index[-indexflag]
        i = (indexnow // blockcol) + 1
        j = (indexnow % blockcol) + 1
        # bbox_list.append([b_h * (j - 1), b_w * (i - 1), b_h * j, b_w * i])  # 目标框的坐标
        indexlist_h.append(indexnow)
        coordinate_list_h.append([i, j])
        indexflag = indexflag + 1
        block_num_h_ori -= 1
        if indexflag >= blockrow * blockcol:
            break

    index, selected_block = FPS(np.array(coordinate_list_h), blocknum_h)
    for i in range(len(index)):
        selected_index_h.append(indexlist_h[index[i]])      # 选中块的索引

    # for i in range(len(selected_index_h)):
    #     indexnow = selected_index_h[i]
    #     i = (indexnow // blockcol) + 1
    #     j = (indexnow % blockcol) + 1
    #     bbox_list.append([b_h * (j - 1), b_w * (i - 1), b_h * j, b_w * i])  # 所选块目标框的坐标

    # 水平方向选块计算水平方向模糊度
    for i in range(len(selected_index_h)):
        indexnow = selected_index_h[i]
        i = (indexnow // blockcol) + 1
        j = (indexnow % blockcol) + 1
        bbox_list_h.append([b_h * (j - 1), b_w * (i - 1), b_h * j, b_w * i])  # 目标框的坐标
        bbox_num1 += 1  # 目标框个数加1
        Block_filted_temp = img_hfilted[(b_h * (i - 1)):(b_h * i), (b_w * (j - 1)):(b_w * j)]
        edges_h_temp = sobel_h_edges[(b_h * (i - 1)):(b_h * i), (b_w * (j - 1)):(b_w * j)]
        edgewidth_h_temp = marziliano_method_h(edges_h_temp, Block_filted_temp)
        total_edges_h_temp = np.count_nonzero(edgewidth_h_temp)
        if total_edges_h_temp > 0:
            edgewidth_h[i - 1, j - 1] = np.sum(edgewidth_h_temp) / total_edges_h_temp   # 计算块的边缘宽度
            edgewidth_h_std[i - 1, j - 1] = np.std(edgewidth_h_temp)                    # 计算块的边缘宽度标准差

        # 去除3_sigma之外的点重新计算均值
        edgewidth_h_temp[edgewidth_h_temp <= (edgewidth_h[i - 1, j - 1] - 3 * edgewidth_h_std[i - 1, j - 1])] = 0
        edgewidth_h_temp[edgewidth_h_temp >= (edgewidth_h[i - 1, j - 1] + 3 * edgewidth_h_std[i - 1, j - 1])] = 0
        total_edges_h_temp = np.count_nonzero(edgewidth_h_temp)
        if total_edges_h_temp > 0:
            edgewidth_h[i - 1, j - 1] = np.sum(edgewidth_h_temp) / total_edges_h_temp   # 计算块的边缘宽度
            edgewidth_h_std[i - 1, j - 1] = np.std(edgewidth_h_temp)                    # 计算块的边缘宽度标准差

        # 画带值的框
        # draw = ImageDraw.Draw(image_color)
        # draw.rectangle([b_h * (j - 1), b_w * (i - 1), b_h * j, b_w * i], fill=None, outline='red', width=10)
        # font = ImageFont.truetype("consola.ttf", 70, encoding="unic")  # 字体
        # text = str(bbox_num1)
        # draw.text([b_h * (j - 1) + 30, b_w * (i - 1) + 30], text, 'maroon', font)
        # text1 = str(round(edgewidth_h[i - 1, j - 1], 2))
        # draw.text([b_h * (j - 1) + 30, b_w * (i - 1) + 120], text1, 'maroon', font=font)
        # del draw


    edgewidth_h_std_nonzero_index = edgewidth_h_std.nonzero()                   # 模糊度方差不为零的块的索引
    edgewidth_h_std_nonzero = edgewidth_h_std[edgewidth_h_std_nonzero_index]    # 模糊度方差不为零的块的模糊度方差
    edgewidth_h_nonzero_index = edgewidth_h.nonzero()                           # 得到edgewidth_h中非零元素的索引值
    edgewidth_h_nonzero = edgewidth_h[edgewidth_h_nonzero_index]                # 得到不为零的块的水平边缘宽度

    std_index = np.argsort(edgewidth_h_std_nonzero)                             # 将方差从小到大排序的索引值
    edgewidth_h_finalblk = edgewidth_h_nonzero[std_index[:blocknum_final]]      # 最终用来计算的小方差块的边缘宽度

    ratio_index_v = np.argsort(ratiolistv)                                      # 块的垂直边缘像素比例从小到大排序的索引
    indexflag = 1           # 索引

    while block_num_v_ori:
        indexnow = ratio_index_v[-indexflag]
        i = (indexnow // blockcol) + 1
        j = (indexnow % blockcol) + 1
        # bbox_list.append([b_h * (j - 1), b_w * (i - 1), b_h * j, b_w * i])  # 目标框的坐标
        indexlist_v.append(indexnow)
        coordinate_list_v.append([i, j])
        indexflag = indexflag + 1
        block_num_v_ori -= 1
        if indexflag >= blockrow * blockcol:
            break

    index, selected_block = FPS(np.array(coordinate_list_v), blocknum_v)
    for i in range(len(index)):
        selected_index_v.append(indexlist_v[index[i]])      # 选中块的索引

    # for i in range(len(selected_index_v)):
    #     indexnow = selected_index_v[i]
    #     i = (indexnow // blockcol) + 1
    #     j = (indexnow % blockcol) + 1
    #     bbox_list.append([b_h * (j - 1), b_w * (i - 1), b_h * j, b_w * i])  # 所选块目标框的坐标

    # 垂直方向选块计算垂直方向模糊度
    for i in range(len(selected_index_v)):
        indexnow = selected_index_v[i]
        i = (indexnow // blockcol) + 1
        j = (indexnow % blockcol) + 1
        bbox_list_v.append([b_h * (j - 1), b_w * (i - 1), b_h * j, b_w * i])  # 目标框的坐标
        bbox_num2 += 1  # 目标框个数加1
        Block_filted_temp = img_vfilted[(b_h * (i - 1)):(b_h * i), (b_w * (j - 1)):(b_w * j)]
        edges_v_temp = sobel_v_edges[(b_h * (i - 1)):(b_h * i), (b_w * (j - 1)):(b_w * j)]
        edgewidth_v_temp = marziliano_method_v(edges_v_temp, Block_filted_temp)
        total_edges_v_temp = np.count_nonzero(edgewidth_v_temp)
        if total_edges_v_temp > 0:
            edgewidth_v[i - 1, j - 1] = np.sum(edgewidth_v_temp) / total_edges_v_temp   # 计算块的垂直边缘宽度
            edgewidth_v_std[i - 1, j - 1] = np.std(edgewidth_v_temp)  # 计算块的边缘宽度标准差

        # 去除3_sigma之外的点重新计算均值
        edgewidth_v_temp[edgewidth_v_temp <= (edgewidth_v[i - 1, j - 1] - 3 * edgewidth_v_std[i - 1, j - 1])] = 0
        edgewidth_v_temp[edgewidth_v_temp >= (edgewidth_v[i - 1, j - 1] + 3 * edgewidth_v_std[i - 1, j - 1])] = 0
        total_edges_v_temp = np.count_nonzero(edgewidth_v_temp)
        if total_edges_v_temp > 0:
            edgewidth_v[i - 1, j - 1] = np.sum(edgewidth_v_temp) / total_edges_v_temp  # 计算块的边缘宽度
            edgewidth_v_std[i - 1, j - 1] = np.std(edgewidth_v_temp)  # 计算块的边缘宽度标准差

        # draw = ImageDraw.Draw(image_color)
        # draw.rectangle([b_h * (j - 1), b_w * (i - 1), b_h * j, b_w * i], fill=None, outline='red', width=10)
        # font = ImageFont.truetype("consola.ttf", 70, encoding="unic")  # 字体
        # text = str(bbox_num2)
        # draw.text([b_h * (j - 1) + 30, b_w * (i - 1) + 30], text, 'maroon', font)
        # text1 = str(round(edgewidth_v[i - 1, j - 1], 2))
        # draw.text([b_h * (j - 1) + 30, b_w * (i - 1) + 120], text1, 'maroon', font=font)
        # del draw

    edgewidth_v_std_nonzero_index = edgewidth_v_std.nonzero()                   # 模糊度方差不为零的块的索引
    edgewidth_v_std_nonzero = edgewidth_v_std[edgewidth_v_std_nonzero_index]    # 模糊度方差不为零的块的模糊度方差
    edgewidth_v_nonzero_index = edgewidth_v.nonzero()                           # 得到edgewidth_h中非零元素的索引值
    edgewidth_v_nonzero = edgewidth_v[edgewidth_v_nonzero_index]                # 得到不为零的块的垂直边缘宽度

    std_index_v = np.argsort(edgewidth_v_std_nonzero)                           # 将方差从小到大排序的索引值
    edgewidth_v_finalblk = edgewidth_v_nonzero[std_index_v[:blocknum_final]]    # 最终用来计算的小方差块的边缘宽度
    # print(edgewidth_h_finalblk, '\n')

    edgewidth_h_final = int(np.round(np.mean(edgewidth_h_finalblk)))            # 计算小方差块的均值作为最终结果
    edgewidth_v_final = int(np.round(np.mean(edgewidth_v_finalblk)))
    blur_pixel = np.sqrt(edgewidth_h_final ** 2 + edgewidth_v_final ** 2)

    # 选块可视化
    # draw = ImageDraw.Draw(image_color)
    # for i in range(len(bbox_list)):
    #     draw.rectangle(bbox_list[i], fill=None, outline='red', width=10)
    # del draw
    # image_color.show()
    # # image_color.save('E:\\传统NR-IQA方法复现\\最远点采样选块(选块更均匀)\\'+ img_name + '_最初20个块(X方向).jpg')
    # image_color.save('E:\\传统NR-IQA方法复现\\实拍图片\\旋转台拍摄\\选块结果1\\' + img_name + '_选中的9个块(X方向).jpg')

    return edgewidth_h_final, edgewidth_v_final, blur_pixel


if __name__ == '__main__':
    # start_time = time.time()
    # # img_name = '21_HUAWEI-Y9_F'
    # # image_path = 'E:\\EIS-OIS\\数据集\\defocused_blurred\\'+img_name+'.jpg'
    # # img_name = 'IMG_20230223_181235_1'
    # # image_path = 'E:\\传统NR-IQA方法复现\\blur_test\\shift\\' + img_name + '.jpg'
    # # img_name = '1 (1)'
    # # image_path = 'E:\\传统NR-IQA方法复现\\实拍图片\\实拍清晰图像\\支架拍摄\\' + img_name + '.jpg'
    # img_name = 'IMG_20230417_111918'
    # image_path = 'E:\\传统NR-IQA方法复现\\实拍图片\\华为4.17\\handheld_less_texture(320)\\' + img_name + '.jpg'
    # print('******' + image_path + '******')
    #
    # edgewidth_h, edgewidth_v, blur_pixel = compute_motion_blur(img_name, image_path, 0)
    # print('x方向模糊像素数是：', edgewidth_h)
    # print('y方向模糊像素数是：', edgewidth_v)
    # print('模值模糊像素数是：', blur_pixel)
    #
    # # edgewidth_135, edgewidth_45, blur_pixel1 = compute_motion_blur(img_name, image_path, 45)
    # #
    # #
    # # print('45度模糊像素数是：', edgewidth_45)
    # # print('135度模糊像素数是：', edgewidth_135)
    # end_time = time.time()
    # run_time = end_time - start_time
    # print('运行时间：', run_time)



    path_name = 'E:\\HUAWEI\\711\\'
    path_list = os.listdir(path_name)
    # header = ['name', 'edgewidth_h', 'edgewidth_v', 'blur_pixel']
    # with open('E:\\传统NR-IQA方法复现\\实拍图片\\华为4.17\\handheld-4\\AutoCapture_20230414_120138(改进后)\\'
    #           'blur_AutoCapture_20230414_120138.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)
    for filename in path_list:
        image = path_name + filename
        img_name, extension = os.path.splitext(filename)
        name_list = img_name.split('-')
        if len(name_list) == 2 and extension == '.jpg':
            print('******'+ image + '******')
            edgewidth_h, edgewidth_v, blur_pixel = compute_motion_blur(img_name, image, 0)

            print('x方向模糊像素数是：', edgewidth_h)
            print('y方向模糊像素数是：', edgewidth_v)
            print('模值模糊像素数是：', blur_pixel)

                # data = [img_name + '.jpg', edgewidth_h, edgewidth_v, blur_pixel]
                # writer.writerow(data)

            # edgewidth_135, edgewidth_45, blur_pixel1 = compute_motion_blur(img_name, image, 45)
            #
            # print('45度模糊像素数是：', edgewidth_45)
            # print('135度模糊像素数是：', edgewidth_135)
            print('\n\n')



    # xlist = []
    # ylist = []
    # yuanlist = []
    # x1list = []
    # y1list = []
    # yuan1list = []

    # 不同长度
    # for ii in range(1, 7):
    #     a = str(ii*5)
    #     # a = str(ii)
    #     image = 'E:/传统NR-IQA方法复现/不同长度运动模糊图像_y方向/48_HONOR-8X_S_len' + a + '_theta90.jpg'
    #     if isinstance(image, str):
    #         image1 = Image.open(image)
    #         image = image1.convert('L')
    #     img = np.array(image, dtype=np.float32)
    #     edgewidth_h, edgewidth_v, blur_pixel = compute_motion_blur(img)
    #     print(ii)
    #     xlist.append(edgewidth_h)
    #     ylist.append(edgewidth_v)
    #     yuanlist.append(blur_pixel)
    #
    # # x1list = [item - min(xlist+ylist) + 5 for item in xlist]
    # # y1list = [item - min(xlist+ylist) for item in ylist]
    # x1list = [item - min(xlist+ylist) for item in xlist]
    # y1list = [item - min(ylist+ylist) + 5 for item in ylist]
    # yuan1list = [math.sqrt(item1 ** 2 + item2 ** 2) for item1, item2 in zip(x1list, y1list)]
    #
    #
    # plt.figure(num='x、y方向及总模糊度曲线')
    # plt.subplot(121)
    # x = [5, 10, 15, 20, 25, 30]
    # plt.plot(x, xlist, label='x方向模糊度', color='c', ls='--', linewidth=3)
    # plt.plot(x, ylist, label='y方向模糊度', color='m', ls='-.', linewidth=3)
    # plt.plot(x, yuanlist, label='总模糊度', color='g', ls='-', linewidth=3)
    # plt.plot(x, x, label='真实值', color='y', ls=':', linewidth=3)
    # plt.xlim([0, 35])
    # plt.ylim([0, max(xlist + ylist) + 10])
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # plt.xlabel('实际模糊像素数')
    # plt.ylabel('模糊像素数')
    # plt.title('实际模糊像素5-30递增时x、y方向及总模糊度(未减最小值)')
    # leg = plt.legend(loc='best')
    # leg_lines = leg.get_lines()
    # leg_text = leg.get_texts()
    # plt.setp(leg_lines, linewidth=2)
    # plt.setp(leg_text, fontsize=12)
    #
    #
    # plt.subplot(122)
    # plt.plot(x, x1list, label='x方向模糊度', color='c', ls='--', linewidth=3)
    # plt.plot(x, y1list, label='y方向模糊度', color='m', ls='-.', linewidth=3)
    # plt.plot(x, yuan1list, label='总模糊度', color='g', ls='-', linewidth=3)
    # plt.plot(x, x, label='真实值', color='y', ls=':', linewidth=3)
    # axes = plt.gca()
    # axes.set_xlim([0, 35])
    # axes.set_ylim([0, max(x1list + y1list) + 10])
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # plt.xlabel('实际模糊像素数')
    # plt.ylabel('模糊像素数')
    # plt.title('实际模糊像素5-30递增时x、y方向及总模糊度(减最小值后)')
    # leg = plt.legend(loc='best')
    # leg_lines = leg.get_lines()
    # leg_text = leg.get_texts()
    # plt.setp(leg_lines, linewidth=2)
    # plt.setp(leg_text, fontsize=12)
    # plt.show()



    # 不同方向
    # for ii in range(7):
    #     a = str(ii*15)
    #     # a = str(ii)
    #     image = 'E:/传统NR-IQA方法复现/不同方向运动模糊图像_20模糊度/238_HONOR-7X_S_len20_theta' + a + '.jpg'
    #     if isinstance(image, str):
    #         image1 = Image.open(image)
    #         image = image1.convert('L')
    #     img = np.array(image, dtype=np.float32)
    #     edgewidth_h, edgewidth_v, blur_pixel = compute_motion_blur(img)
    #     print(ii)
    #     xlist.append(edgewidth_h)
    #     ylist.append(edgewidth_v)
    #     yuanlist.append(blur_pixel)
    #
    # # x1list = [item - min(xlist+ylist) + 5 for item in xlist]
    # # y1list = [item - min(xlist+ylist) for item in ylist]
    # x1list = [item - min(xlist+ylist) for item in xlist]
    # y1list = [item - min(xlist+ylist) for item in ylist]
    # yuan1list = [math.sqrt(item1 ** 2 + item2 ** 2) for item1, item2 in zip(x1list, y1list)]
    #
    #
    # plt.figure(num='x、y方向及总模糊度曲线')
    # plt.subplot(121)
    # x = [0, 15, 30, 45, 60, 75, 90]
    # plt.plot(x, xlist, label='x方向模糊度', color='c', ls='--', linewidth=3)
    # plt.plot(x, ylist, label='y方向模糊度', color='m', ls='-.', linewidth=3)
    # plt.plot(x, yuanlist, label='总模糊度', color='g', ls='-', linewidth=3)
    # # plt.plot(x, x, label='真实值', color='y', ls=':', linewidth=3)
    # plt.xlim([-10, 100])
    # plt.ylim([0, max(xlist + ylist) + 10])
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    # plt.xlabel('模糊角度')
    # plt.ylabel('模糊像素数')
    # plt.title('模糊角度从0-90°递增时x、y方向及总模糊度(未减最小值)')
    # leg = plt.legend(loc=1)
    # leg_lines = leg.get_lines()
    # leg_text = leg.get_texts()
    # plt.setp(leg_lines, linewidth=2)
    # plt.setp(leg_text, fontsize=12)
    #
    #
    # plt.subplot(122)
    # plt.plot(x, x1list, label='x方向模糊度', color='c', ls='--', linewidth=3)
    # plt.plot(x, y1list, label='y方向模糊度', color='m', ls='-.', linewidth=3)
    # plt.plot(x, yuan1list, label='总模糊度', color='g', ls='-', linewidth=3)
    # # plt.plot(x, x, label='真实值', color='y', ls=':', linewidth=3)
    # axes = plt.gca()
    # axes.set_xlim([-10, 100])
    # axes.set_ylim([0, max(x1list + y1list) + 10])
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    # plt.xlabel('模糊角度')
    # plt.ylabel('模糊像素数')
    # plt.title('模糊角度从0-90°递增时x、y方向及总模糊度(减最小值后)')
    # leg = plt.legend(loc=1)
    # leg_lines = leg.get_lines()
    # leg_text = leg.get_texts()
    # plt.setp(leg_lines, linewidth=2)
    # plt.setp(leg_text, fontsize=12)
    # plt.show()







    # sigmalist = [1, 2, 3, 4, 5, 6]
    #
    # # 散焦模糊不同sigma
    # for ii in range(6):
    #     a = str(sigmalist[ii])
    #     image = 'E:/传统NR-IQA方法复现/不同sigma高斯模糊图像_size15/3_HUAWEI-NOVA-LITE_S_size15_sigma' + a + '.jpg'
    #     if isinstance(image, str):
    #         image1 = Image.open(image)
    #         image = image1.convert('L')
    #     img = np.array(image, dtype=np.float32)
    #     edgewidth_h, edgewidth_v, blur_pixel = compute_motion_blur(img)
    #     print(ii)
    #     xlist.append(edgewidth_h)
    #     ylist.append(edgewidth_v)
    #     yuanlist.append(blur_pixel)
    #
    # # x1list = [item - min(xlist) + 5 for item in xlist]
    # # y1list = [item - min(ylist) for item in ylist]
    # x1list = [item - min(xlist+ylist) + 2 for item in xlist]
    # y1list = [item - min(xlist+ylist) + 2 for item in ylist]
    # yuan1list = [math.sqrt(item1 ** 2 + item2 ** 2) for item1, item2 in zip(x1list, y1list)]
    #
    # plt.figure(num='x、y方向及总模糊度曲线')
    # plt.subplot(121)
    # x = sigmalist
    # plt.plot(x, xlist, label='x方向模糊度', color='c', ls='--', linewidth=3)
    # plt.plot(x, ylist, label='y方向模糊度', color='m', ls='-.', linewidth=3)
    # plt.plot(x, yuanlist, label='总模糊度', color='g', ls='-', linewidth=3)
    # # plt.plot(x, x, label='真实值', color='y', ls=':', linewidth=3)
    # plt.xlim([0, 7])
    # plt.ylim([0, max(xlist + ylist) + 10])
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # plt.xlabel('高斯模糊σ')
    # plt.ylabel('模糊像素数')
    # plt.title('高斯模糊σ从1-6递增时x、y方向及总模糊度(未减最小值)')
    # leg = plt.legend(loc=0)
    # leg_lines = leg.get_lines()
    # leg_text = leg.get_texts()
    # plt.setp(leg_lines, linewidth=2)
    # plt.setp(leg_text, fontsize=12)
    #
    # plt.subplot(122)
    # plt.plot(x, x1list, label='x方向模糊度', color='c', ls='--', linewidth=3)
    # plt.plot(x, y1list, label='y方向模糊度', color='m', ls='-.', linewidth=3)
    # plt.plot(x, yuan1list, label='总模糊度', color='g', ls='-', linewidth=3)
    # # plt.plot(x, x, label='真实值', color='y', ls=':', linewidth=3)
    # axes = plt.gca()
    # axes.set_xlim([0, 7])
    # axes.set_ylim([0, max(x1list + y1list) + 10])
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # plt.xlabel('高斯模糊σ')
    # plt.ylabel('模糊像素数')
    # plt.title('高斯模糊σ从1-6递增时x、y方向及总模糊度(减最小值后)')
    # leg = plt.legend(loc=0)
    # leg_lines = leg.get_lines()
    # leg_text = leg.get_texts()
    # plt.setp(leg_lines, linewidth=2)
    # plt.setp(leg_text, fontsize=12)
    # plt.show()


