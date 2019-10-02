#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri March 15 20:05:23 2019

@author: 2020shatgiskessell
"""

import sys
sys.path.insert(0, '/Users/2020shatgiskessell/anaconda3/pkgs/opencv-3.3.1-py36h60a5f38_1/lib/python3.6/site-packages/opencv3')
import cv2
import numpy as np
import timeit
import time
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import warnings
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

sys.path.append('/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Mole_Detector_1_2/')
from compute_stats import compute_stats
from Blob import Blob
#ヽ(≧Д≦)ノ
warnings.filterwarnings("ignore")

img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/mole10.0.jpg")

#SKIN DETECTOR CNN
#https://github.com/HasnainRaz/Skin-Segmentation-TensorFlow

def draw_blobs (image, blobs):
    #draw blobs
    for blob in blobs:
        cv2.circle(image, (int(blob.x),int(blob.y)), 5, (0,255,0), 1)

def draw_blobs_coordinates (image, xs, ys):
    #draw blobs
    for i in range (len(xs)):
        #the non-validated blob coordinates are NAN so a try catch statetment is necessary
        try:
            cv2.circle(image, (int(xs[i]),int(ys[i])), 5, (0,255,0), 1)
        except ValueError:
            pass

def imshow_components(labels, img):
    #https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python?rq=1
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    #cv2.imshow('og.png', img)

    #cv2.imshow('original', img)

    #cv2.waitKey(0)

def blob_detector (img, x_i, y_i):
    og = img.copy()
    img = cv2.Canny(img, 100, 200)

    blobs = []
    #get connected components of fg
    #print ("calculating connected components...")
    output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    #print ("calculating statistics...")
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
    #print (centroids)
    #print(np.where(labels == 2))

    #imshow_components(labels, og)
    #print ("calculating connected components properties...")
    with concurrent.futures.ThreadPoolExecutor(5) as executor:
        future_moles = {executor.submit(compute_stats, num_labels,labels, stats, centroids, img, x_i, y_i, i, og):i for i in range(num_labels)}
        for future in concurrent.futures.as_completed(future_moles):
            found_blobs = future.result()
            if found_blobs != None:
                blobs.extend(found_blobs)
    return blobs

def sobel(img):
    blurred = cv2.GaussianBlur(img, (3,3),0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)


    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def Laplacian_sharpen2(img):
    img = np.float32(img)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(im_gray,(5,5),0)

    # img_lap = cv2.Laplacian(img_blur, cv2.CV_32F, 3)
    # _, img_lap= cv2.threshold(img_lap, 127, 255, cv2.THRESH_BINARY)
    # #img_lap = cv2.normalize(img_lap,None, 0, 255, cv2.NORM_MINMAX)
    # img_lap = img_lap.astype('uint8')
    #
    # subtracted = im_gray - img_lap
    # #img_lap = img_lap.astype('uint8')
    # subtracted = subtracted.astype('uint8')

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened = cv2.filter2D(img_blur, -1, kernel)
    sharpened = cv2.normalize(sharpened,None, 0, 255, cv2.NORM_MINMAX)
    sharpened = sharpened.astype('uint8')
    return sharpened

def Laplacian_sharpen(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(im_gray,(5,5),0)
    #img_blur2 = cv2.GaussianBlur(im_gray,(5,5),0)

    img_lap = cv2.Laplacian(img_blur, cv2.CV_16S, 3)
    im_gray = np.float32(im_gray)
    subtracted = im_gray - img_lap
    subtracted = np.clip(subtracted, 0, 255)
    #subtract_gus = img_blur2 - img_blur

    #img_lap = img_lap.astype('uint8')
    subtracted = subtracted.astype('uint8')
    return subtracted

def mse(x, y):
    return np.linalg.norm(x - y)

def get_centroid(cluster):
  cluster_ary = np.asarray(cluster)
  centroid = cluster_ary.mean(axis = 0)
  return centroid

def plot_mole_coordinates (blobs):
    x = []
    y = []
    #get x and y coordinates for every blob
    for blob in blobs:
        x.append(blob.x)
        y.append(blob.y)
    #plt.scatter(x, y)
    #plt.show()
    return x,y

def get_skin(img, sharpened):
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

    sharpened = cv2.bitwise_and(sharpened, sharpened, mask = skinMask)

    return sharpened

#When you filter connected components by size, it also gets rid of bigger moles_recognized
def main (img, id_):
    #resize image to 400x300 (4:3 aspect ratio)
    start = timeit.default_timer()
    img = cv2.imread(img)
    img = cv2.resize(img, (400,300))

    #draw blobs on image
    img1 = img.copy()
    #img2 = img.copy()
    #img3 = img.copy()
    img4 = img.copy()

    #create sliding window to detect initial blobs
    #blobs_found = sliding_window(img1, 5, [32,32])
    #print ("sharpening image...")
    sharpened1 = Laplacian_sharpen(img1)
    #sharpened2 = Laplacian_sharpen2(img1)
    #cv2.imwrite("sharpened1.png", sharpened)
    #sharpened = sobel(img1)
    sharpened = get_skin(img1, sharpened1)
    blobs = blob_detector(sharpened,0,0)
    #plot the mole coordinates
    #print ("plotting mole coordinates...")
    x,y = plot_mole_coordinates(blobs)

    #remove false positive blobs using DBSCAN clustering - PLAY AROUND WITH PARAMETERS
    #x2,y2, n_moles = validate_blob3(x,y,30)

    draw_blobs(img1, blobs)
    #draw_blobs_coordinates(img4, x2,y2)
    stop = timeit.default_timer()

    print('Time: ', stop - start)

    #----------------------------------
    cv2.imshow('all', img1)
    cv2.imwrite('all1.png', img1)
    print (len(blobs))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return x,y, len(blobs), img1
