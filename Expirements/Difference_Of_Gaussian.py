#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:54:09 2019

@author: 2020shatgiskessell
"""

import cv2

img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/blob3419.081694690656.jpg")

def dog (img):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(imgray,(1,1),0)
    blur2 = cv2.GaussianBlur(imgray,(3,3),0)
    laplacian = cv2.Laplacian(blur1 - blur2,cv2.CV_64F)
    return laplacian

result = dog (img)
cv2.imshow("result", result)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()