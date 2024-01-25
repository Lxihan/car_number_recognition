import cv2
import numpy as np
import os
path='./image/10.jpg'
image = cv2.imread(path)
filename_without_extension = os.path.splitext(os.path.basename(path))[0]
# 将图像转换为HSV色彩空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#定义车牌颜色范围
lower_bound = np.array([int(210/2), 50, int(0.5*255)])
upper_bound = np.array([int(225/2), 255, 255])
#根据颜色范围确定掩膜
mask=cv2.inRange(hsv_image,lower_bound,upper_bound)

# 形态学操作，提取车辆区域
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(mask, kernel, iterations=1)
# 对原始图像和掩膜进行位运算，提取车牌区域
cv2.imshow("mask",mask)
plate_area = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("plate_area",plate_area)
# 对掩膜进行形态学操作
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(mask, kernel, iterations=1)
# 寻找轮廓
contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 过滤和筛选轮廓，选择可能的车牌区域
plate_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    area = cv2.contourArea(contour)
    if aspect_ratio > 2.5 and aspect_ratio < 4 and area>4000:
        plate_contours.append(contour)

# 绘制车牌区域的边界框
for contour in plate_contours:
    x, y, w, h = cv2.boundingRect(contour)
    plate_image = image[y:y + h, x:x + w]  # Crop the plate region
    output_filename = f'{filename_without_extension}_plate.jpg'
    cv2.imwrite(output_filename, plate_image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 显示定位后的图像
cv2.imshow('License Plate Localization', image)

hsv_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask1=cv2.inRange(hsv_image1,lower_bound,upper_bound)

cv2.imshow("mask1", mask1)
cv2.waitKey(0)