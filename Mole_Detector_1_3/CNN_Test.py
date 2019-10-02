import cv2
import numpy as np

from keras.models import load_model

img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/ANN_Images/roi330.png",0)
img = cv2.resize(img, (8,8))
img = np.expand_dims(img, axis=2)
img = np.expand_dims(img, axis=0)

model = load_model('my_mole_model.h5')
pred = model.predict(img)
print (pred.round())
