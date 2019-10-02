#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:15:29 2019

@author: 2020shatgiskessell
"""

import cv2
import numpy as np
img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/mole2.0_all_blobs/blob89200.70073540197.png")
circle = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/circle.png")

class Blob ():
    def __init__ (self, x = None,y = None,area = None, roi = None, matrix = None, radius = None, color = None):
        self.x = x
        self.y = y
        self.area = area
        self.roi = roi
        self.matrix = matrix
        self.radius = radius
        self.color = color

def draw_blobs_coordinates (image, xs, ys, radiuses):
    #draw blobs
    print ("drawn " + str(len(xs)) + " blobs")
    for i in range (len(xs)):
        #the non-validated blob coordinates are NAN so a try catch statetment is necessary
        try:
            cv2.circle(image, (int(xs[i]),int(ys[i])), radiuses[i], (0,255,0), 1)
        except ValueError:
            pass
        
def imshow_components(labels):
    #https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python/46442154
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()

def compute_formfactor(radius, area):
    #computre circle perimeter
    perimeter = np.pi * radius * 2
    #compute circularity
    circularity = (np.pi*4*area)/np.power(perimeter,2)
    return circularity

def compute_roundness (diameter, radius, area):
    roundness = (4*area)/(np.pi*np.power((diameter),2))
    return roundness


def shape_similarity(roi1, roi2):
    #calculate hu moment
    img_moments1 = cv2.moments(roi1)
    img_moments2 = cv2.moments(roi2)
    
    hu1 = cv2.HuMoments(img_moments1).flatten()
    hu2 = cv2.HuMoments(img_moments2).flatten()
    #log scale transformation to normalize hu values
    distance = 0
    for i in range(0,7):
        hu1[i] = -np.sign(hu1[i]) * np.log10(np.abs(hu1[i]))
        hu2[i] = -np.sign(hu2[i]) * np.log10(np.abs(hu2[i]))
        #match contours - THIS IS JUST A DEFUALT EQUATION!!!
        inv_diff = np.abs((1/hu2[i]) - (1/hu1[i]))
        diff = np.abs(float(hu1[i]) - float(hu2[i]))
        distance += inv_diff
    return distance



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
        
        #compute statistics
        width = stats[i,cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        radius = (height + width)/2
        area = stats[i, cv2.CC_STAT_AREA]
        #remove everything except for component i to create isolated component matrix
        component_matrix = [[1 if m== i else 0 for m in marker] for marker in labels]
        #get connected component roi
        
        #TODO 
        #WRONG X Y- THESE R CENTROID X Y NOT CORNER X Y POSITIONS!!!!!
        x_l = stats[i,cv2.CC_STAT_LEFT]
        y_l = stats[i,cv2.CC_STAT_TOP]
        roi = img[y_l: y_l+height, x_l:x_l+width]
        #print ("area: " + str(area) + ", circularity: " + str(circularity))
        #print ("height: " + str(height) + ", width: " + str(width))
        #----------------OTHER CIRCULARITY MEASURES-------------------------------------------------------------------
        
        radius = (height + width)/2
        formfactor = compute_formfactor (radius, area)
        if height > width:
            roundness = compute_roundness (height, radius, area)
            aspect_ratio = height/width
        else:
            roundness = compute_roundness (width, radius, area)
            aspect_ratio = width/height
        
        print ("radius: " + str(radius) + ", formfactor: " + str(formfactor) + ", roundness: " + str(roundness) + ", aspect ratio: " + str(aspect_ratio))

        #----------------SHAPE SIMILARITY IS A SILLY WAY TO DO THIS-------------------------------------------------------------------
#        #difference = shape_similarity(roi, circle)
#        d2 = cv2.matchShapes(roi,circle,cv2.CONTOURS_MATCH_I2,0)
#        print ("cv difference: " + str(d2))
#        if d2 < 0.02:
#            blob = Blob()
#            blob.radius = radius
#            blob.x = x
#            blob.y = y
#            blob.matrix = component_matrix
#            blobs.append(blob)
#        #----------------CIRCULARITY MEASUE IS WRONG-------------------------------------------------------------------
#        #compute circularity
#        radius = (height + width)/2
#        circularity = compute_circularity (radius)
#        #check if component is circular
        if 1.2 > roundness > 0.9 and aspect_ratio < 2 and formfactor > 0.2:
            print ("recognized blob")
            blob = Blob()
            blob.radius = radius
            blob.x = x
            blob.y = y
            blob.matrix = component_matrix
            blobs.append(blob)
        cv2.imshow("component", roi)
        cv2.waitKey(0)
#        #----------------CIRCULARITY MEASUE IS WRONG-------------------------------------------------------------------
    #imshow_components(labels)
    return blobs


img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(img, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

circle = cv2.cvtColor(circle,cv2.COLOR_BGR2GRAY)
circle = 255 - circle
#_, circle = cv2.threshold(circle, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

components = blob_detector (thresh)
xs =[]
ys = []
radiuses = []
for blob in components:
    xs.append(blob.x)
    ys.append(blob.y)
    radiuses.append(int(blob.radius))
draw_blobs_coordinates (img, xs, ys, radiuses)

#cv2.imshow("circle", circle)

cv2.imshow("contours", thresh)
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()