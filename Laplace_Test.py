import numpy as np
import cv2

img = cv2.imread("/Users/2020shatgiskessell/Downloads/99399.jpg")


def Laplacian_sharpen(img):
    #img = np.float32(img)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(im_gray,(5,5),0)
    #img_blur2 = cv2.GaussianBlur(im_gray,(5,5),0)

    img_lap = cv2.Laplacian(img_blur, cv2.CV_32F, 3)
    #_, img_lap= cv2.threshold(img_lap, 0, 255, cv2.THRESH_BINARY)

    #subtracted = im_gray - img_lap
    #subtracted = np.clip(subtracted, 0, 255)


    img_lap = img_lap.astype('uint8')
    print (im_gray)
    print (img_lap)
    #subtracted = subtracted.astype('uint8')

    cv2.imshow("sub",img_lap )
    cv2.imshow("og",img )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_lap

#Laplacian_sharpen(img)
arr = np.ones((5,5))*(1/25)
print (arr)
