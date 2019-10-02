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
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import warnings
#ヽ(≧Д≦)ノ
warnings.filterwarnings("ignore")

start = timeit.default_timer()

class Blob ():
    def __init__ (self, x = None,y = None,area = None, roi = None, matrix = None, radius = None, color = None):
        self.x = x
        self.y = y
        self.area = area
        self.roi = roi
        self.matrix = matrix
        self.radius = radius
        self.color = color

img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/mole7.jpg")

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

def compute_formfactor(radius, area):
    #computre circle perimeter
    perimeter = np.pi * radius * 2
    #compute circularity
    circularity = (np.pi*4*area)/np.power(perimeter,2)
    return circularity

def compute_roundness (diameter, radius, area):
    roundness = (4*area)/(np.pi*np.power((diameter),2))
    return roundness

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

    #cv2.imshow('labeled.png', labeled_img)
    #cv2.imshow('original', img)

    cv2.waitKey()

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
    component_centroids = {}

    #imshow_components(labels, og)
    #print ("calculating connected components properties...")
    for i in range (1, num_labels):
        #start = timeit.default_timer()
        #get component centroid coordinates
        x,y = centroids[i]

        #compute statistics
        width = stats[i,cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if height > 20:
            continue
        radius = (height + width)/2
        area = stats[i, cv2.CC_STAT_AREA]

        #these are the top left x, y coordinates = ONLY TO BE USED FOR GETTING ROI
        x_l = stats[i,cv2.CC_STAT_LEFT]
        y_l = stats[i,cv2.CC_STAT_TOP]
        #stop = timeit.default_timer()
        #print('Time to calculate properties: ', stop - start)


        #start = timeit.default_timer()
        #remove everything except for component i to create isolated component matrix
        #component_matrix = [[1 if m== i else 0 for m in marker] for marker in labels]
        #get connected component roi
        roi = img[y_l: y_l+height, x_l:x_l+width]

        #stop = timeit.default_timer()
        #print('Time to create cm and roi: ', stop - start)

        #----------------MEASURES-------------------------------------------------------------------
        #radius = (height + width)/2

        start = timeit.default_timer()
        radius = np.sqrt((area/np.pi))
        formfactor = compute_formfactor (radius, area)
        if height > width:
            roundness = compute_roundness (height, radius, area)
            aspect_ratio = height/width
        else:
            roundness = compute_roundness (width, radius, area)
            aspect_ratio = width/height

        stop = timeit.default_timer()
        #print('Time to calculate heuristic properties: ', stop - start)

        #print ("radius: " + str(radius) + ", formfactor: " + str(formfactor) + ", roundness: " + str(roundness) + ", aspect ratio: " + str(aspect_ratio))
        # print ("area: " + str(area))
        # print("Roundness: " + str(roundness))
        # print("aspect_ratio: " + str(aspect_ratio))
        # print("formfactor: " + str(formfactor))
        # print ("(" + str(x) + ","+str(y)+")")
        # print ("\n")

        #WRONG X,Y COORDINATE!

        if roundness > 0.2 and 0.9 < aspect_ratio < 3 and 1.1 >= formfactor > 0.9:
            blob = Blob()
            blob.radius = radius
            blob.x = x+x_i
            blob.y = y+y_i
            #blob.matrix = component_matrix
            blob.roi = roi
            blobs.append(blob)
        #cv2.imshow("roi", roi)
        #cv2.waitKey(0)
    #print ("Found all blobs")
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
    #subtract_gus = img_blur2 - img_blur

    #img_lap = img_lap.astype('uint8')
    subtracted = subtracted.astype('uint8')
    #subtract_gus = subtract_gus.astype('uint8')

    #subtract_gus_blurred = cv2.medianBlur(subtract_gus,3)
    #subtract_gus_blurred = cv2.GaussianBlur(subtract_gus_blurred,(7,7),0)
    # kernel = np.ones((2,2),np.uint8)
    # eroded = cv2.erode(subtract_gus,kernel,1)
    # dilated = cv2.dilate(eroded,kernel,1)
    #
    # edges = cv2.Canny(subtract_gus_blurred,100,200)
    # kernel_c = np.ones((5,5),np.uint8)
    # closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_c)
    #closing = cv2.GaussianBlur(closing,(7,7),0)

    # cv2.imshow("original",img)
    # cv2.imshow("subtract_gus_blurred",subtract_gus_blurred)
    # cv2.imshow("edges",edges)

    return subtracted

def sliding_window(image, stepSize, windowSize):
    blobs_found = []
    #blobs = []
    # slide a window across the image
    #image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #blur1 = cv2.GaussianBlur(img,(3,3),0)
    #_, thresh = cv2.threshold(image, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # identify moles on  current window
            #blobs.append(identify_moles(image[y:y + windowSize[1], x:x + windowSize[0]]))
            winodw = image[y:y + windowSize[1], x:x + windowSize[0]]
            sharpened = Laplacian_sharpen(winodw)
            blobs = blob_detector(sharpened,x,y)
            blobs_found.extend(blobs)
            #move window and animate everything
            # clone = image.copy()
            # cv2.rectangle(clone, (x, y), (x + windowSize[0], y + windowSize[1]), (0, 255, 0), 2)
            # cv2.imshow("Window", clone)
            # cv2.imshow("identified moles",diff)
            # cv2.waitKey(1)
            # time.sleep(0.025)
    return blobs_found

def mse(x, y):
    return np.linalg.norm(x - y)

def get_centroid(cluster):
  cluster_ary = np.asarray(cluster)
  centroid = cluster_ary.mean(axis = 0)
  return centroid

def validate_blob3 (x, y, n):
    #https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    validated_x = []
    validated_y = []

    #turn coordinates into 2d array
    coordinates = np.vstack((x, y)).T
    #kmeans clustering
    #kmeans = KMeans(n_clusters=n, random_state=0).fit(coordinates)
    clustered = DBSCAN(eps=5, min_samples=5).fit(coordinates)

    core_samples_mask = np.zeros_like(clustered.labels_, dtype=bool)
    core_samples_mask[clustered.core_sample_indices_] = True
    labels = clustered.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #n_noise_ = list(labels).count(-1)

    #print ("number of moles identified: " + str(n_clusters_))

    #plot the clusters
    #plt.scatter(coordinates[:,0],coordinates[:,1], c=clustered.labels_, cmap='rainbow')
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    #loop through every cluster
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        #get cluster label
        class_member_mask = (labels == k)
        #plot only the clusters
        xy = coordinates[class_member_mask & core_samples_mask]
        #xy[0] is the first cluster
        #get centroid for cluster
        centroid = get_centroid(xy)
        #add centroid coordinates to array
        validated_x.append(centroid[0])
        validated_y.append(centroid[1])
    return validated_x,validated_y, n_clusters_

def is_it_brown (rgb):
    red = rgb[2]
    green = rgb[1]
    blue = rgb[0]

    #check different thresholds brown falls in - Somehow this works
    if blue >= 128:
        return False
    if np.abs(red - green) > 100:
        return False
    if red+green+blue < 30:
        return False
    else:
        return True

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

def main (img, id_):
    #resize image to 400x300 (4:3 aspect ratio)
    #img = cv2.resize(img, (400,300))

    #draw blobs on image
    img1 = img.copy()
    #img2 = img.copy()
    #img3 = img.copy()
    img4 = img.copy()

    #create sliding window to detect initial blobs
    img1 = get_skin(img1, img1)
    blobs = sliding_window(img1, 4, (32,32))

    #plot the mole coordinates
    x,y = plot_mole_coordinates(blobs)

    #remove false positive blobs using DBSCAN clustering - PLAY AROUND WITH PARAMETERS
    x2,y2, n_moles = validate_blob3(x,y,30)

    draw_blobs(img1, blobs)
    draw_blobs_coordinates(img4, x2,y2)
    stop = timeit.default_timer()

    print('Time: ', stop - start)

    #----------------------------------
    cv2.imwrite('all1.png', img4)
    # cv2.imshow('sharpened1', sharpened1)
    # cv2.imshow('sharpened2', sharpened2)
    #cv2.imshow('dbscan', img4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return x,y, len(blobs), img4

main (img, 1)
