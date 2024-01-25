from os import listdir
import operator
import numpy as np
import cv2
import os
def cvshow(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
path="./image/7.jpg"
simage = cv2.imread(path,0);
s = cv2.imread(path)
filename_without_extension = os.path.splitext(os.path.basename(path))[0]
colorImage = s.copy()
hsv=cv2.cvtColor(s,cv2.COLOR_BGR2HSV)
#cvshow('hsv',hsv)
sum=0
for i in range(simage.shape[0]):
    for j in range(simage.shape[1]):
       sum+=simage[i][j]
th = sum/(simage.shape[0]*simage.shape[1])
gaussian = cv2.GaussianBlur(simage, (3, 3), 0, 0, cv2.BORDER_DEFAULT)#高斯滤波处理
#cvshow('gaussian',gaussian)
ret,th1 = cv2.threshold(gaussian,th-10,255,cv2.THRESH_BINARY)#二值化处理
canny = cv2.Canny(th1,80, 120)
cvshow('canny',canny)
#lines = cv2.HoughLines(canny, 1, np.pi/180, 180)
#进行霍夫直线检测
lines = cv2.HoughLines(canny, 1, np.pi/180, 155)
height, width = canny.shape[:2]
# 存储竖直线的角度范围

horizontal_angle_range = (0, 10)
horizontal_lines=[]
print(lines.size)
for line in lines:
    rho, theta = line[0]
    # 计算线段的斜率（角度）
    #angle = np.degrees(theta)
    angle = theta * 180 / np.pi
    # 通过斜率判断是否为竖直线段
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    #cv2.line(s, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if abs(x1 - x2) > 20:
        #print(x1, y1, x2, y2)
        horizontal_lines.append((x1, y1, x2, y2))
    #cv2.line(s, (x1, y1), (x2, y2), (0, 0, 255), 2)
cvshow('s',s)
prev_bottom = 0
count = 0
chosen_lines = []
a, b, c, d, e, f, g, h = 0, 0, 0, 0, 0, 0, 0, 0
#寻找符合条件的直线
for x1, y1, x2, y2 in sorted(horizontal_lines, key=lambda x: x[1], reverse=True):
    if count == 0:
        prev_bottom = min(y1, y2)
        a, b, c, d = x1, y1, x2, y2
    else:
        if prev_bottom - min(y1, y2) > 40:
            e, f, g, h = x1, y1, x2, y2
            break
    count += 1
print(a,b,c,d)
print(e,f,g,h)
#low=min(b,d)
#high=max(f,h)
low=int((b+d)/2)
high=int((f+h)/2)
cv2.line(s, (a, low), (c, low), (0, 0, 255), 2)
cv2.line(s, (e, high), (g,high), (0, 0, 255), 2)
x, y, w, h = cv2.boundingRect(np.array([(a, low), (c, low), (e, high), (g, high)]))
border_area = s[y:y + h, x:x + w]
#将得到的上下边界的图片裁剪输出
output_filename = f'{filename_without_extension}_plate.jpg'
cv2.imwrite(output_filename, border_area)
#cv2.line(s, (0, y2), (width, y2), (0, 0, 255), 2)
cvshow('s',s)