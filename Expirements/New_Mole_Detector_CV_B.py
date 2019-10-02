#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:01:35 2019

@author: 2020shatgiskessell
"""

import sys
sys.path.insert(0, '/Users/2020shatgiskessell/anaconda3/pkgs/opencv-3.3.1-py36h60a5f38_1/lib/python3.6/site-packages/opencv3')
import cv2
import numpy as np
import timeit
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import warnings
#ヽ(≧Д≦)ノ
warnings.filterwarnings("ignore")

#start = timeit.default_timer()

class Blob ():
    def __init__ (self, x = None,y = None,area = None, roi = None, matrix = None, radius = None, color = None):
        self.x = x
        self.y = y
        self.area = area
        self.roi = roi
        self.matrix = matrix
        self.radius = radius
        self.color = color
        

img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/mole2.0.jpg")
#img2 = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/mole4.1.jpg")

template = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/gaus2d.jpg")

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

def display_blob_info (blobs_found):
#    for b in blobs_found:
#        print ("(" + str(b.x) + "," + str(b.y) + ")")
#        print ("size: "+ str(b.size))
    cv2.imshow("ROI", blobs_found[20].roi)

def sliding_window(image, stepSize, windowSize):
    blobs_found = []
    #blobs = []
    # slide a window across the image
    image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #blur1 = cv2.GaussianBlur(img,(3,3),0)
    _, thresh = cv2.threshold(image, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # identify moles on  current window
            #blobs.append(identify_moles(image[y:y + windowSize[1], x:x + windowSize[0]]))\
            blobs = identify_moles(thresh[y:y + windowSize[1], x:x + windowSize[0]],x,y) 
            blobs_found.extend(blobs)
#--------------------------------------------------------------------------------------------------------
            #move window and animate everything
#            clone = image.copy()
#            cv2.rectangle(clone, (x, y), (x + windowSize[0], y + windowSize[1]), (0, 255, 0), 2)
#            cv2.imshow("Window", clone)
#            cv2.imshow("identified moles",identified_moles)
#            cv2.waitKey(1)
#            time.sleep(0.025)
#--------------------------------------------------------------------------------------------------------
    return blobs_found
def compute_circularity (radius):
    #compute circle area
    area = np.pi * np.power(radius,2)
    #computre circle perimeter
    perimeter = np.pi * radius * 2
    #compute circularity
    circularity = (np.pi*4*area)/np.power(perimeter,2)
    return circularity

def blob_detector (img):
    blobs = []
    #get connected components of fg
    output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
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
    for centroid in centroids:
        #get x,y coordinates of centroids
        x,y = centroid
        #get corrosposding component for each centroid
        component = labels[int(y)][int(x)]
        #print(component)
        #save in dictionary
        component_centroids[component] = list(centroid)

    for i in range (0, num_labels):
        #get component centroid coordinates
        try:
            x,y = component_centroids.get(i)
        except Exception as e:
            print (e)
        x = int(x)
        y = int(y)
        print (str(x) + "," + str(y))
        
        #compute circularity
        width = stats[i,cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        radius = (height + width)/2
        
        
        circularity = compute_circularity (radius)
        #print ("width: " + str(width) + " ,height: " + str(height) + " ,radius: " + str(radius))
        if 1.1 > circularity > 0.9:
            blob = Blob()
            blob.radius = radius
            blob.x = x
            blob.y = y
            #remove everything except for component i to create isolated component matrix
            component_matrix = [[1 if m== i else 0 for m in marker] for marker in labels]
            blob.matrix = component_matrix
            blobs.append(blob)
    #imshow_components(labels)
    return blobs


def identify_moles (image_window, x, y):
    #create blob detctor
    blobs = blob_detector(image_window)
    #get blob information and store it 
    for blob in blobs:
        blob.x = blob.x + x
        blob.y = blob.y + y
        #print ("rect is: (" + str(int(Blob.y -radius)) + "," + str(int(Blob.y +radius)) + "," + str(int( Blob.x-radius))+ "," + str(int( Blob.x+radius)) + ")")
        offset = 2
        blob.roi = image_window[int(blob.y -blob.radius)-offset:int(blob.y +blob.radius) + offset,int(blob.x-blob.radius) - offset:int(blob.x+blob.radius)+offset]
        #blob color
        blob.color = find_dominant_color(image_window[int(blob.y -blob.radius):int(blob.y +blob.radius),int(blob.x-blob.radius):int(blob.x+blob.radius)])
    return blobs
#--------------------------------------------------------------------------------------------------------
def mse(x, y):
    return np.linalg.norm(x - y)

def find_dominant_color (img):
    #https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
    try:
        #reshape image
        pixels = np.float32(img.reshape(-1, 3))
        
        #number of colors
        n_colors = 5
        
        #K-means clusteering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        
        #get morst frequent color
        dominant = palette[np.argmax(counts)]
        return dominant
    
    except Exception:
        return None

def validate_blob (blobs, template, cutoff):
    validated_blobs = []
    for blob in blobs:
        #cv2.imwrite(os.path.join("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Mole_ROIs", "roi"+str(i)+".jpg"), blob.roi)
        try:
            #turn image to grayscale
            gray = 255 - cv2.cvtColor(blob.roi,cv2.COLOR_BGR2GRAY)
            #get dimensions of image
            height, width = gray.shape 
            #reshape template image
            resized_template = cv2.resize(template, (width, height))
            #turn image to grayscale
            gray_template = 255 - cv2.cvtColor(resized_template,cv2.COLOR_BGR2GRAY)

            #mean squared error
            mse_ = mse(gray, gray_template)

            #only keep moles that have a error lower then cutoff
            if int(mse_) < cutoff:
                validated_blobs.append(blob)
            #invert
            #cv2.imwrite(os.path.join("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Mole_ROIs_Binary", "roi"+str(i)+".jpg"), gray)
            #i = i+1
        except TypeError:
            pass
    return validated_blobs

def validate_blob2 (blobs, cutoff):
    #---------------------------------------------------------------
    #https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
    #---------------------------------------------------------------
    
    #https://softwareengineering.stackexchange.com/questions/129892/find-all-points-within-a-certain-distance-of-each-other
    #1. Divide the space into a regular lattice of cubes. The the length of the side of each cube should be half the minimum distance between points.
    #2. For the first point, see which cube it is in. 
    #   Add a flag to the cube that contains the point to indicate that that cube contains a point of that color.
    #3. For the next point, see which cube is in. If none of the adjacent cubes are flagged with a point of that color, flag the cube with the color of the point like in (2). 
    #   Otherwise throw the point away.
    #4. If you still have unprocessed points, go to 3.
    
    recognized = []
    #traverse array 
    for  blob in blobs:
        for blob2 in blobs:
        #compare closeness of point coordinates
            blob_c = [blob.x, blob.y]
            blob2_c = [blob2.x, blob2.y]
            #multiply distances so that it is not rounded to 0
            distance_ = distance.euclidean (blob_c, blob2_c) * 10000000
            #check if distance is less than cutoff point 
            if distance_ <  cutoff and distance_ != 0:
                #print( distance_)
                #add clustered blobs to array
                recognized.append(blob)
                blobs.remove(blob)
                blobs.remove(blob2)
                break

    print (str(len(recognized)) + " : " + str(len(blobs)))
    return recognized
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
#    plt.scatter(coordinates[:,0],coordinates[:,1], c=clustered.labels_, cmap='rainbow')  

#-------------------------------------------------------------------------------------------
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
        #plot all validated blobs
        #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 #markeredgecolor='k', markersize=14)

        #plot none validated blobs
#        xy = coordinates[class_member_mask & ~core_samples_mask]
#        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                 markeredgecolor='k', markersize=6)
    #plt.title('Estimated number of clusters: %d' % n_clusters_)
    #plt.show()
#-------------------------------------------------------------------------------------------
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



def main (img):
    #resize image to 400x300 (4:3 aspect ratio)
    img = cv2.resize(img, (400,300))
    
    #remove everything in array
    #blobs_found.clear()

    #draw blobs on image
    img1 = img.copy()
    #img2 = img.copy()
    #img3 = img.copy()
    img4 = img.copy()
    
    #create sliding window to detect initial blobs
    blobs_found = sliding_window(img1, 4, [32,32])
    
    #remove false positive blobs using guassian structure similarity
    #validated_blobs = validate_blob(blobs_found, template,1800)
    
    #remove false positive blobs using euclidean distance
    #validated_blobs2 = validate_blob2(blobs_found, 60)
    
    
    #plot the mole coordinates
    x,y = plot_mole_coordinates(blobs_found)
    #plot_mole_coordinates(validated_blobs)
    #plot_mole_coordinates(validated_blobs2)
    
    #remove false positive blobs using DBSCAN clustering - PLAY AROUND WITH PARAMETERS
    x2,y2, n_moles = validate_blob3(x,y,30)
    
    #display blob properties
    #display_blob_info(blobs_found)
    
    
    
    draw_blobs(img, blobs_found)
    #draw_blobs(img2, validated_blobs)
    #draw_blobs(img3, validated_blobs2)
    draw_blobs_coordinates(img4, x2,y2)
    #----------------------------------
    cv2.imshow('all', img)
    #cv2.imshow('input', img1)
    #-----------------------------------
    #cv2.imshow('gaussian_recognized', img2)
    #cv2.imshow('euclidean_recognized', img3)
    
    #cv2.imshow('DBSCAN_clustered', img4)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return x2,y2, n_moles

main(img)

#x1,y1,n_moles1 = main (img)
#x2,y2,n_moles2 = main (img2)
#print ("You have  " + str(n_moles2 - n_moles1) + " new moles")
#stop = timeit.default_timer()
#
#print('Time: ', stop - start)  
