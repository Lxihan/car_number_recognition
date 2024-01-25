import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def cvshow(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def invert_binary_image(binary_image):
    # 将黑色设置为 1
    binary_image = binary_image.astype(np.uint8)
    binary_image = 255 - binary_image
    return binary_image
path = "5_plate.jpg"
filename_without_extension = os.path.splitext(os.path.basename(path))[0]
plate_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cvshow('image',plate_image)
_,binary_image=cv2.threshold(plate_image, 100, 255, cv2.THRESH_BINARY)
cvshow('binary',binary_image)
vertical_projection = np.sum(binary_image, axis=0)
# 创建图像
fig, ax = plt.subplots()

    # 绘制垂直投影
ax.plot(vertical_projection)

    # 设置标题
ax.set_title("Vertical projection")

    # 显示图像
plt.show()
# 设定阈值来确定左右边界,
threshold = 0.8 * np.max(vertical_projection)
left_boundary = np.min(np.where(vertical_projection > threshold))
right_boundary = np.max(np.where( vertical_projection>threshold))
threshold1=0.8*np.max(vertical_projection)
if left_boundary<10:#如果为反转的情况，我们就将像素值最大值变为最小值即可。
    inverted_binary_image = invert_binary_image(plate_image)
    vertical_projection=np.sum(inverted_binary_image,axis=0)
    fig, ax = plt.subplots()

    # 绘制垂直投影
    ax.plot(vertical_projection)

    # 设置标题
    ax.set_title("Vertical projection")

    # 显示图像
    plt.show()
    threshold = 0.8 * np.max(vertical_projection)
    threshold1 = 0.8 * np.max(vertical_projection)
    left_boundary = np.min(np.where(vertical_projection > threshold))
    right_boundary = np.max(np.where(vertical_projection > threshold))
plate_with_boundaries = cv2.imread(path)
#cv2.line(plate_with_boundaries, (left_boundary, 0), (left_boundary, plate_image.shape[0]), (0, 255, 0), 2)
#cv2.line(plate_with_boundaries, (right_boundary, 0), (right_boundary, plate_image.shape[0]), (0, 255, 0), 2)
#cvshow('Plate with Boundaries', plate_with_boundaries)
line_positions = [left_boundary, right_boundary]

# 裁剪原图
cropped_plate = plate_with_boundaries[:, line_positions[0]:line_positions[1]]
# 显示裁剪后的图像，并裁剪出来，用于后续的提取字符
cvshow('Cropped Plate', cropped_plate)
output_filename = f'{filename_without_extension}_1.jpg'
cv2.imwrite(output_filename, cropped_plate)