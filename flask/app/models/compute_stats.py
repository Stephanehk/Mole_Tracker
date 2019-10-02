import numpy as np
import cv2
import os

from Blob import Blob

def compute_formfactor(radius, area):
    #computre circle perimeter
    perimeter = np.pi * radius * 2
    #compute circularity
    circularity = (np.pi*4*area)/np.power(perimeter,2)
    return circularity

def compute_roundness (diameter, radius, area):
    roundness = (4*area)/(np.pi*np.power((diameter),2))
    return roundness

def error_metric (xs, ys,x1,y1,x2,y2):
    #https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    error = 0
    for x,y in zip(xs,ys):
        numerator = np.abs(((y2-y1)*x) - ((x2-x1)*y) + ((x2*y1) - (y2*x1)))
        denomonator = np.sqrt(np.power(y2-y1,2) + np.power(x2-x1,2))
        distance = np.power(numerator/denomonator,2)
        error += distance

    return error

def compute_lobf(roi,id):
    #find contours
    _, roi_cnt, h = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rows,cols = roi.shape[:2]
    for cnt in roi_cnt:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        #print (box)
        #(vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line.
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        #print (str(vx) + "," + str(vy) + "," + str(x) + "," + str(y))
        #get y cords from normal and contour cords
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        #draw line
        cv2.line(roi,(cols-1,righty),(0,lefty),(0,255,0),1)
        #print ("("+str(cols-1) + "," + str(righty) + ") -> (0," + str(lefty) + ") \n")
        #caluclate error
        xs = box[:,0]
        ys = box[:,1]
        error = error_metric (xs, ys,cols-1,righty,0,lefty)
        #cv2.imwrite("roi" + str(error) +" --"+str(id) + ".png", roi)

        return (cols-1,righty),(0,lefty), error


def compute_stats (num_labels,labels, stats, centroids, img, x_i, y_i, i, og):
    blobs = []
    #start = timeit.default_timer()
    #get component centroid coordinates
    x,y = centroids[i]
    #compute statistics
    # cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
    # cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
    # cv2.CC_STAT_WIDTH The horizontal size of the bounding box
    # cv2.CC_STAT_HEIGHT The vertical size of the bounding box
    # cv2.CC_STAT_AREA The total area (in pixels) of the connected component

    width = stats[i,cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    if height > 20:
        return None,None,None,None,None, None
    radius = (height + width)/2
    area = stats[i, cv2.CC_STAT_AREA]

    #these are the top left x, y coordinates = ONLY TO BE USED FOR GETTING ROI
    x_l = stats[i,cv2.CC_STAT_LEFT]
    y_l = stats[i,cv2.CC_STAT_TOP]

    #compute line
    #slope, y_int= compute_lobf(x,y,x_l,y_l)

    #stop = timeit.default_timer()
    #print('Time to calculate properties: ', stop - start)

    #remove everything except for component i to create isolated component matrix
    #get connected component roi
    roi = og[y_l-1: y_l+height+1, x_l-1:x_l+width+1]
    #stop = timeit.default_timer()
    #print('Time to create cm and roi: ', stop - start)

    #----------------MEASURES-------------------------------------------------------------------
    #radius = (height + width)/2

    #compute more statistics related to roundness
    radius = np.sqrt((area/np.pi))
    formfactor = compute_formfactor (radius, area)
    bounding_box_area_ratio = area/(height*width)
    if height > width:
        roundness = compute_roundness (height, radius, area)
        aspect_ratio = height/width
    else:
        roundness = compute_roundness (width, radius, area)
        aspect_ratio = width/height

    #print('Time to calculate heuristic properties: ', stop - start)
    # if x >300 and y < 150:
    #     print ("(" + str(x) + ","+str(y)+") -> area ratio: " + str(area/(height*width)))
    #print ("(" + str(x) + ","+str(y)+") -> " + "radius: " + str(radius) + ", formfactor: " + str(formfactor) + ", roundness: " + str(roundness) + ", aspect ratio: " + str(aspect_ratio))
    #print ("(" + str(x) + ","+str(y)+")")
    #print ("\n")

    #calculates line of best fit and error
    try:
        cord1, cord2, error = compute_lobf(roi, x_l*y_l)
        x1,y1 = cord1
        x2,y2 = cord2
    except TypeError:
        print ("cant calculate line of best fit")
        error = 0

    cv2.imwrite(os.path.join("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/ANN_Images_Fiverr" , "roi19" + str(i) + ".png"), roi)

    #COMMENT OUT WHEN COLLECTING ANN DATA!!!!!
    #if the error is below 16, (the line of best fit closely matches connect component) return none
    if error < 16:
        return None,None,None,None,None, None


    # print ("(" + str(x1) + ","+str(y1)+")")
    # print ("(" + str(x2) + ","+str(y2)+")")
    # print ("\n")

    #next step: calculate residuals and do some sort of analysis

    #try  and bounding_box_area_ratio >= 0.5
    if roundness > 0.2 and 0.9 < aspect_ratio < 3 and 1.1 >= formfactor > 0.9:
        blob = Blob()
        blob.radius = radius
        blob.x = x+x_i
        blob.y = y+y_i
        #blob.matrix = component_matrix
        blob.roi = roi
        blobs.append(blob)


    return blobs

#Compactness of a BLOB is defined as the ratio of the BLOB’s area to the area of the bounding box. This can be used to distinguish compact BLOBs from noncompact ones.
#Center of the bounding box is a fast approximation of the center of mass. In mathematical terms the center of the bounding box, (xbb,ybb) is calculated as
#Compute color of blob
#   - look into HSV
#   - sum pixel intensities
