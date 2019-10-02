#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:20:48 2019

@author: 2020shatgiskessell
"""

import cv2
import numpy as np

def get_skin(img):
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel)

    skinMask = cv2.GaussianBlur(skinMask, (3,3),0)
    
    img = cv2.bitwise_and(img, img, mask = skinMask)

    return img
