import numpy as np
import openslide
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ImageAlignment


# 打开wsi文件
slide1 = openslide.open_slide('D:/Files/wsi/A091_PD-L1_HE.svs')
slide2 = openslide.open_slide('D:/Files/wsi/A091_PD-L1_V.svs')
# img1.save('D:/Files/wsi/17-62087-HE.jpg', quality=95)
# img2.save('D:/Files/wsi/17-62087-PD_L1.jpg', quality=95)


VIEW_WINDOWS_SIZE = [1000, 1000]
SEGMENTATION_SIZE = [1500, 1500]
# rot = img1.rotate(ROTATION_DEG)
# rot.reduce()

# 加载出原始的两个图像
img1 = slide1.read_region((0, 0), 3, slide1.level_dimensions[3])
img2 = slide2.read_region((0, 0), 3, slide1.level_dimensions[3])

# 划分截取区域
slide1_region = (36788, 23079)
slide2_region = (20467, 24159)
LEVEL = 2

sec1 = slide1.read_region(slide1_region, LEVEL, VIEW_WINDOWS_SIZE)
sec2 = slide2.read_region(slide2_region, LEVEL, VIEW_WINDOWS_SIZE)
print("dowm", slide2.level_dimensions)

# 鼠标点击输入
class Pointer:
    def __init__(self, point):
        self.point = point
        self.xs = list(point.get_xdata())
        self.ys = list(point.get_ydata())
        self.cid = point.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print(event)
        if event.inaxes != self.point.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.point.set_data(self.xs, self.ys)
        # print(self.xs, self.ys)
        self.point.figure.canvas.draw()

    def give_coordinates(self):
        x = self.xs
        y = self.ys
        return x, y


vx = int(VIEW_WINDOWS_SIZE[0] / (slide1.level_downsamples[3]/slide1.level_downsamples[LEVEL]))
vy = int(VIEW_WINDOWS_SIZE[1] / (slide2.level_downsamples[3]/slide2.level_downsamples[LEVEL]))
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_title('click to add points')
fig = plt.figure(figsize=(20, 20))
axs1 = fig.add_subplot(221)
axs1.imshow(img1)
axs1.set_title("A091_PD-L1_HE")

axs1.add_patch(patches.Rectangle((int(slide1_region[0] / (slide1.level_downsamples[3]/slide1.level_downsamples[0])),
                                  int(slide1_region[1] / (slide1.level_downsamples[3]/slide1.level_downsamples[0]))),
                                 vx, vy, linewidth=1, edgecolor='r', facecolor='none'))
axs2 = fig.add_subplot(222)
axs2.imshow(img2)
axs2.add_patch(patches.Rectangle((int(slide2_region[0] / (slide1.level_downsamples[3]/slide1.level_downsamples[0])),
                                  int(slide2_region[1] / (slide1.level_downsamples[3]/slide1.level_downsamples[0]))),
                                 vx, vy, linewidth=1, edgecolor='r', facecolor='none'))
axs2.set_title("A091_PD-L1_V")
axs3 = fig.add_subplot(223)
axs3.imshow(sec1)
axs4 = fig.add_subplot(224)
axs4.imshow(sec2)


point1, = axs3.plot([], [], linestyle="-", marker="o", color="r", markersize=3)
x1, y1 = Pointer(point1).give_coordinates()
point2, = axs4.plot([], [], linestyle="-", marker="o", color="g", markersize=3)
x2, y2 = Pointer(point2).give_coordinates()
plt.show()
print(x1, y1)
print(x2, y2)

# point2, = axs2.plot([], [], linestyle="none", marker="o", color="r")
# pointer2 = Pointer(point2).give_coordinates()

#  #键盘输入不同的点坐标
# ref_coordinate1 = []
# ref_coordinate2 = []
# num_of_ref = int(input('How many point you want to put?'),10)
#
#
# for i in range(num_of_ref):
#     ref_coordinate1.append(input("Enter your coordinate {} separated by ',' ".format(i+1)).split(','))
# print(ref_coordinate1)
#
#
# for i in range(num_of_ref):
#     ref_coordinate2.append(input("Enter your coordinate {} separated by ',' ".format(i+1)).split(','))
# print(ref_coordinate2)

if not len(x1) == len(x2):
    print("please make sure that each image has given the same number of points")

else:
    print("Points accepted")


# 对旋转角度进行计算
def transformation(refx1, refy1, refx2, refy2, patch_dim, rotation_only=False, matrix_mode=False):

    p_x, p_y = patch_dim

    if matrix_mode:  # 通过opencv的仿射矩阵进行计算。 计算平移，旋转和缩放， 使用3个对应坐标点。

        cord_set1 = np.float32([refx1[0], refy1[0]],
                            [refx1[1], refy1[1]],
                            [refx1[2], refy1[2]])

        cord_set2 = np.float32([refx2[0], refy2[0]],
                            [refx2[1], refy2[1]],
                            [refx2[2], refy2[2]])

        matrix = cv2.getAffineTransform(cord_set1, cord_set2)

        return matrix
    else:  # 自主开发算法，只计算平移和旋转，可以使用多个坐标点

        rotationl = []

        for i in range(len(refx1) - 1):

            a1 = np.array([(refx1[i + 1] - refx1[i]), (refy1[i + 1] - refy1[i])])

            a2 = np.array([(refx2[i + 1] - refx2[i]), (refy2[i + 1] - refy2[i])])

            angle = np.arccos(a1.dot(a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)))

            rotationl.append(angle)

        angle = sum(rotationl) / len(rotationl)
        rotation_angle = np.degrees(angle)

        crefx1 = [i - int(p_x / 2) for i in refx1]
        crefy1 = [i - int(p_y / 2) for i in refy1]
        crefx2 = [i - int(p_x / 2) for i in refx2]
        crefy2 = [i - int(p_y / 2) for i in refy2]

        xtrans = []
        ytrans = []

        for i in range(len(refx1)):
            cord_new = np.array([crefx2[i], crefy2[i]]).transpose().__matmul__([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
            xtrans.append(cord_new[0] - crefx1[i])
            ytrans.append(cord_new[1] - crefy1[i])

        xtran = sum(xtrans)/len(xtrans)
        ytran = sum(ytrans)/len(ytrans)

        # plt.subplot(121).imshow(sec1)
        # plt.subplot(121).plot(refx1, refy1, linestyle="-", marker="o", color="r", markersize=3)
        # plt.subplot(122).imshow(sec2.rotate(rotation_angle))
        # plt.subplot(122).plot(nrefx2, nrefy2, linestyle="-", marker="o", color="r", markersize=3)
        # plt.show()

        if rotation_only:
            return rotation_angle
        else:
            return rotation_angle, xtran, ytran  # 计算得出两个切片的旋转和x与y轴上的平移关系


# 生成旋转对准后的图像
sec2full = slide2.read_region((int(slide2_region[0] - 0.5 * VIEW_WINDOWS_SIZE[0] * (slide2.level_downsamples[2] / slide2.level_downsamples[0])),
                               int(slide2_region[1] - 0.5 * VIEW_WINDOWS_SIZE[1] * (slide2.level_downsamples[2] / slide2.level_downsamples[0]))),
                              LEVEL, (int(VIEW_WINDOWS_SIZE[0] * 2), int(VIEW_WINDOWS_SIZE[1] * 2)))
x1 = np.array(x1)
y1 = np.array(y1)
x2 = np.array(x2)
y2 = np.array(y2)
rotation, x_p, y_p = ImageAlignment.manuel_registration(x1, y1, x2, y2, VIEW_WINDOWS_SIZE)
print(rotation)
a = int(0.5 * VIEW_WINDOWS_SIZE[0]) + x_p
b = int(0.5 * VIEW_WINDOWS_SIZE[0]) + y_p
c = int(1.5 * VIEW_WINDOWS_SIZE[0]) + x_p
d = int(1.5 * VIEW_WINDOWS_SIZE[0]) + y_p

sec2_r = sec2full.rotate(rotation).crop((a, b, c, d))


# 显示完成对准后的图像
fig2 = plt.figure(figsize=(10, 10))
axs1 = fig2.add_subplot(221)
axs1.imshow(sec1)
axs1.set_title("A091_PD-L1_HE")
# axs1.add_patch(patches.Rectangle((int(slide1_region[0] / (slide1.level_downsamples[3]/slide1.level_downsamples[0])),
#                                   int(slide1_region[1] / (slide1.level_downsamples[3]/slide1.level_downsamples[0]))),
#                                  vx, vy, linewidth=1, edgecolor='r', facecolor='none'))
axs2 = fig2.add_subplot(222)
axs2.imshow(sec2_r)
# axs2.add_patch(patches.Rectangle((int(slide2_region[0] / (slide1.level_downsamples[3]/slide1.level_downsamples[0])),
#                                   int(slide2_region[1] / (slide1.level_downsamples[3]/slide1.level_downsamples[0]))),
#                                  vx, vy, linewidth=1, edgecolor='r', facecolor='none'))

sec1 = np.asarray(sec1)
sec2_r = np.asarray(sec2_r)
axs2.set_title("A091_PD-L1_V")
axs3 = fig2.add_subplot(223)
image1 = cv2.addWeighted(sec1, 0.7, sec2_r, 0.3, 0)
axs3.imshow(image1)
# axs3.plot(x1, y1, linestyle="-", marker="o", color="r", markersize=3)
image2 = cv2.addWeighted(sec1, 0.5, sec2_r, 0.5, 0)
axs4 = fig2.add_subplot(224)
axs4.imshow(image2)
plt.show()