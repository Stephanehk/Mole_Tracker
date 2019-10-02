#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:36:09 2019

@author: 2020shatgiskessell
"""
import cv2
import timeit
import numpy as np

import Mole_Detector_1_0
import Mole_Detector_1_1
import Mole_Detector_1_2

# img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/mole4.0.jpg")
# img2 = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/mole4.1.jpg")

def draw_blobs_coordinates (image, xs, ys):
    #draw blobs
    for i in range (len(xs)):
        #the non-validated blob coordinates are NAN so a try catch statetment is necessary
        try:
            cv2.circle(image, (int(xs[i]),int(ys[i])), 5, (0,255,0), 1)
        except ValueError:
            pass

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

def find_moles(img, img2, function):
    function = eval(function)
    start = timeit.default_timer()
    img_skin = get_skin(img)
    img2_skin = get_skin(img2)

    x1,y1,number_of_moles1 = function.main(img_skin,1)
    draw_blobs_coordinates (img, x1, y1)

    x2,y2,number_of_moles2 = function.main(img2_skin,2)
    draw_blobs_coordinates (img2, x2, y2)

    print ("before: " + str(number_of_moles1) + ", after: " + str(number_of_moles2))

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    cv2.imwrite("first"+str(number_of_moles1)+"time:"+str(stop - start)+".png", img)
    cv2.imwrite("second"+str(number_of_moles2)+"time:"+str(stop - start)+".png", img2)

def find_moles_singular(img, function):
    function = eval(function)
    start = timeit.default_timer()
    img_skin = get_skin(img)

    x1,y1,number_of_moles1 = function.main(img_skin,1)
    draw_blobs_coordinates (img, x1, y1)

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    cv2.imwrite("first"+str(number_of_moles1)+"time:"+str(stop - start)+".png", img)

def main2 (image_paths1,image_paths2, function):
    for i in range (len(image_paths1)):
        path1 = image_paths1[i]
        path2 = image_paths2[i]
        img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/"+path1+".jpg")
        img2 = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/"+path2+".jpg")
        find_moles(img, img2, function)

def main1 (image_paths_singular, function):
    for path1 in image_paths_singular:
        img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/"+path1+".jpg")
        find_moles_singular(img, function)

image_paths1 = ["mole2.0", "mole2.1", "mole3.0", "mole4.0"]
image_paths2 = ["mole2.1", "mole2.2", "mole3.1", "mole4.1"]
image_paths_singular = ["mole5", "mole6","mole7","mole8"]

main1(image_paths_singular, "Mole_Detector_1_1")
main2(image_paths1,image_paths2, "Mole_Detector_1_1")
