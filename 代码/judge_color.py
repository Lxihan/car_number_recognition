import cv2
import numpy as np
import os
# 载入图像
def cvshow(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
dir_path='./image'
out='./out'
out_path = 'C:\\Users\\I\\Desktop\\202111000411-林训浩\\实验结果'
file_names = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
print(file_names)
for file in file_names:
    a=file[0]
    if file[0]=='1' and file[1]=='0':
        a+=file[1]
    print(a)
    print(dir_path+'/'+file)
    path=dir_path+'/'+file
    image = cv2.imread(path)
    enhanced_image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image1 = cv2.equalizeHist(gray_image)
    binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]
    #cvshow('image',image)
    #cvshow('enhance',enhanced_image)
    #cvshow('gray',gray_image)
    #cvshow('equalize',enhanced_image1)
    #cvshow('binary',binary_image)
    cv2.imwrite(out +'enhanced_image_'+file, enhanced_image)
    cv2.imwrite(out + '/gray_' + file, gray_image)
    cv2.imwrite(out + '/equalize_' + file, enhanced_image1)
    cv2.imwrite(out + '/binary_' + file, binary_image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #cvshow('hsv',hsv_image)
    cv2.imwrite(out + '/hsv_' + file, hsv_image)
    lower_bound = np.array([int(210 / 2), 50, int(0.5 * 255)])
    upper_bound = np.array([int(225 / 2), 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    #cvshow('mask',mask)
    cv2.imwrite(out + '/mask_' + file,mask)
    white_area = cv2.countNonZero(mask)
    print(f"白色区域的面积: {white_area} 像素")
    if white_area<2000:
        cvshow('green',image)
    else:
        cvshow('blue',image)