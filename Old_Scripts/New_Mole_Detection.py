#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:48:48 2018

@author: 2020shatgiskessell
"""
import cv2
import numpy as np

#IN LATER VERSIONS MAKE SURE IMAGE IS ALWAYS BINARIZED OR GRAYSCALE!!
img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/test3.png")

##use blob detectiom
#def identify_moles (image):
#    #create blob detector and pass image through
#    detector = cv2.SimpleBlobDetector_create()
#    keypoints = detector.detect(image)
#    
#    #draw blobs
#    blank = np.zeros((1,1))
#    blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
#    
#    return blobs

#use blob detectiom with parameters
def identify_moles (image):
    edges = cv2.Canny(image,100,200)
    #erode then dialate image
#    kernel = np.ones((2,2),np.uint8)
#    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    #create blob detector and pass image through
    params = cv2.SimpleBlobDetector_Params()
    
    #only get dark blobs
    params.blobColor = 0
    
     #filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0
    
    #filter by area
    params.minArea = 12
    
    #filter by threshold
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(edges)

    #draw blobs
    blank = np.zeros((1,1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    return blobs

#ALGO IDEA TO GET RID OF EXTRANEOUS MOLES DETECTED:
    #1. add every blob detected to an array
    #2. add corropsonding color value to array
    #3. sort both arrays based on color value (so darkest color valued blob is first)
    #4. compare every other blobs color value to this blobs color valye
    #5. get rid of the blobs that are far in color valye (ie: they are not moles)

image_with_blob = identify_moles(img)
total = 0
for i in image_with_blob:
    total = total + 1
print ("Total: " + str(total))

cv2.imshow('identified moles', image_with_blob)

cv2.imshow('original image', img)
#cv2.imwrite("idnetified moles.png", image_with_blobs)

cv2.waitKey(0)
cv2.destroyAllWindows()