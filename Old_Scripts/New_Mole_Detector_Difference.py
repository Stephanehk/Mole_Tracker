#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:47:05 2019

@author: 2020shatgiskessell
"""

import cv2
import numpy as np
import time

img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/mole2.0.jpg")
template = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/gaus2d.jpg")


def sliding_window(image, stepSize, windowSize):
    #blobs = []
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            #print ("excecuting")
            # identify moles on  current window
            #blobs.append(identify_moles(image[y:y + windowSize[1], x:x + windowSize[0]]))\
            roi = image[y:y + windowSize[1], x:x + windowSize[0]]
            img = template_match (roi,template, 0.5)
#--------------------------------------------------------------------------------------------------------
            #move window and animate everything
            clone = image.copy()
            cv2.rectangle(clone, (x, y), (x + windowSize[0], y + windowSize[1]), (0, 255, 0), 2)
            cv2.imshow("Window", img)
            cv2.imshow("identified moles",clone)
            cv2.waitKey(1)
            time.sleep(0.025)
#--------------------------------------------------------------------------------------------------------


def template_match (img, template, threshold):
    gray = 255 - cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #get dimensions of image
    height, width = gray.shape 
    #reshape template image
    resized_template = cv2.resize(template, (width, height))
    #turn image to grayscale
    gray_template =cv2.cvtColor(resized_template,cv2.COLOR_BGR2GRAY)
#    
    #template mtching
    res = cv2.matchTemplate(gray,gray_template,cv2.TM_CCOEFF_NORMED)
    print (res)
    #get values within threshold
    loc = np.where( res >= threshold)
    
    for pt in zip(*loc[::-1]): 
        cv2.rectangle(img, pt, (pt[0] + width, pt[1] + height), (0,255,255), 2)
  
    return img
    # Show the final image with the matched area. 
#    cv2.imshow('Detected',img) 
    

sliding_window(img,4,[10,10])

cv2.waitKey(0)
cv2.destroyAllWindows()