import sys
sys.path.insert(0, '/Users/2020shatgiskessell/anaconda3/pkgs/opencv-3.3.1-py36h60a5f38_1/lib/python3.6/site-packages/opencv3')
import cv2
import numpy as np
import warnings
#ヽ(≧Д≦)ノ
warnings.filterwarnings("ignore")

class Blob ():
    def __init__ (self, x = None,y = None,area = None, roi = None, matrix = None, radius = None, color = None):
        self.x = x
        self.y = y
        self.area = area
        self.roi = roi
        self.matrix = matrix
        self.radius = radius
        self.color = color

def imshow_components(labels):
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

def blob_detector (img, x_i, y_i):
    img = cv2.Canny(img, 100, 200)

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

    for i in range (1, num_labels):
        #get component centroid coordinates
        x,y = centroids[i]

        #compute statistics
        width = stats[i,cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        radius = (height + width)/2
        area = stats[i, cv2.CC_STAT_AREA]

        #these are the top left x, y coordinates = ONLY TO BE USED FOR GETTING ROI
        x_l = stats[i,cv2.CC_STAT_LEFT]
        y_l = stats[i,cv2.CC_STAT_TOP]

        #remove everything except for component i to create isolated component matrix
        component_matrix = [[1 if m== i else 0 for m in marker] for marker in labels]
        #get connected component roi
        roi = img[y_l: y_l+height, x_l:x_l+width]

        #----------------MEASURES-------------------------------------------------------------------
        #radius = (height + width)/2
        radius = np.sqrt((area/np.pi))
        formfactor = compute_formfactor (radius, area)
        if height > width:
            roundness = compute_roundness (height, radius, area)
            aspect_ratio = height/width
        else:
            roundness = compute_roundness (width, radius, area)
            aspect_ratio = width/height

        #print ("radius: " + str(radius) + ", formfactor: " + str(formfactor) + ", roundness: " + str(roundness) + ", aspect ratio: " + str(aspect_ratio))
        print ("area: " + str(area))
        print("Roundness: " + str(roundness))
        print("aspect_ratio: " + str(aspect_ratio))
        print("formfactor: " + str(formfactor))
        print ("(" + str(x) + ","+str(y)+")")
        print ("\n")

        #WRONG X,Y COORDINATE!

        if roundness > 0.2 and 0.9 < aspect_ratio < 1.5 and 1.1 >= formfactor > 0.9:
            print ("is circle")
            blob = Blob()
            blob.radius = radius
            blob.x = x+x_i
            blob.y = y+y_i
            blob.matrix = component_matrix
            blob.roi = roi
            blobs.append(blob)
        #cv2.imshow("roi", roi)
        #cv2.waitKey(0)
    return blobs

def Laplacian_sharpen(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(im_gray,(3,3),0)
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

def draw_blobs (image, blobs):
    #draw blobs
    for blob in blobs:
        cv2.circle(image, (int(blob.x),int(blob.y)), 5, (0,255,0), 1)

def hough_circles(img):
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=1, param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#img2 = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/test4.png")
img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Test_Images/mole4.0.jpg")
subtracted = Laplacian_sharpen(img)
blobs = blob_detector (subtracted, 0, 0)
draw_blobs (img, blobs)

# blobs = blob_detector (diff, 0, 0)
# blob_img = img.copy()
# draw_blobs (blob_img, blobs)
cv2.imshow("orig",img)
# cv2.imshow("diff",diff)
# cv2.imshow("subtracted",subtracted)

cv2.waitKey(0)

cv2.destroyAllWindows()
