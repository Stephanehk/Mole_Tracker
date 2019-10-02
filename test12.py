import cv2
import numpy as np
import sys

img1 = cv2.imread("/Users/2020shatgiskessell/Downloads/city_test_img.jpg",0)
img2 = cv2.imread("/Users/2020shatgiskessell/Downloads/city_test_img_2.jpg",0)

bf = cv2.BFMatcher()
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
print (des1)
