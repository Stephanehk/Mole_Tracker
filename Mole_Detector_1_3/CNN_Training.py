import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
#
# is_mole = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0,1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,1, 0, 1, 1,1,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,0,1,1,0,0,1,0, 0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0]
# #
# rois = ['roi13.png', 'roi15.png', 'roi14.png', 'roi17.png', 'roi16.png', 'roi19.png', 'roi18.png', 'roi111.png', 'roi110.png', 'roi113.png', 'roi112.png', 'roi114.png', 'roi116.png', 'roi115.png', 'roi121.png', 'roi119.png', 'roi122.png', 'roi123.png', 'roi125.png', 'roi126.png', 'roi127.png', 'roi128.png', 'roi22.png', 'roi23.png', 'roi24.png', 'roi25.png', 'roi26.png', 'roi28.png', 'roi27.png', 'roi29.png', 'roi210.png', 'roi213.png', 'roi212.png', 'roi211.png', 'roi214.png', 'roi216.png', 'roi215.png', 'roi218.png', 'roi217.png', 'roi219.png', 'roi221.png', 'roi222.png', 'roi220.png', 'roi223.png', 'roi224.png', 'roi225.png', 'roi227.png', 'roi229.png', 'roi228.png', 'roi230.png', 'roi231.png', 'roi232.png', 'roi234.png', 'roi233.png', 'roi235.png', 'roi236.png', 'roi238.png', 'roi237.png', 'roi239.png', 'roi241.png', 'roi240.png', 'roi242.png', 'roi243.png', 'roi245.png', 'roi244.png', 'roi246.png', 'roi247.png', 'roi248.png', 'roi249.png', 'roi250.png', 'roi34.png', 'roi33.png', 'roi32.png', 'roi36.png', 'roi38.png', 'roi35.png', 'roi37.png', 'roi312.png', 'roi310.png', 'roi311.png', 'roi313.png', 'roi315.png', 'roi314.png', 'roi316.png', 'roi317.png', 'roi318.png', 'roi319.png', 'roi321.png', 'roi320.png', 'roi322.png', 'roi323.png', 'roi327.png', 'roi326.png', 'roi325.png', 'roi324.png', 'roi328.png', 'roi329.png', 'roi331.png', 'roi330.png', 'roi332.png', 'roi333.png', 'roi334.png', 'roi335.png', 'roi337.png', 'roi338.png', 'roi336.png', 'roi339.png', 'roi340.png', 'roi341.png', 'roi343.png', 'roi342.png', 'roi344.png', 'roi345.png', 'roi346.png', 'roi348.png', 'roi349.png', 'roi347.png', 'roi350.png', 'roi351.png', 'roi352.png', 'roi353.png', 'roi356.png', 'roi354.png', 'roi358.png', 'roi359.png', 'roi355.png', 'roi357.png', 'roi361.png', 'roi360.png', 'roi363.png', 'roi364.png', 'roi362.png', 'roi365.png', 'roi366.png', 'roi367.png', 'roi369.png', 'roi371.png', 'roi368.png', 'roi373.png', 'roi370.png', 'roi374.png', 'roi375.png', 'roi376.png', 'roi377.png', 'roi378.png', 'roi372.png', 'roi379.png', 'roi380.png', 'roi381.png', 'roi382.png', 'roi383.png', 'roi385.png', 'roi384.png', 'roi386.png', 'roi387.png', 'roi389.png', 'roi388.png', 'roi390.png', 'roi391.png', 'roi392.png', 'roi393.png', 'roi394.png', 'roi396.png', 'roi398.png', 'roi397.png', 'roi399.png', 'roi3100.png', 'roi3102.png', 'roi3104.png', 'roi3105.png', 'roi3106.png', 'roi3101.png', 'roi3108.png', 'roi3107.png', 'roi3109.png', 'roi3110.png', 'roi3111.png', 'roi3113.png', 'roi3112.png', 'roi3114.png', 'roi3115.png', 'roi3119.png', 'roi3118.png', 'roi3116.png', 'roi3117.png', 'roi3120.png', 'roi3121.png', 'roi3122.png', 'roi3125.png', 'roi3124.png', 'roi3123.png', 'roi3126.png', 'roi3127.png', 'roi3129.png', 'roi3133.png', 'roi3131.png', 'roi3128.png', 'roi3136.png', 'roi3132.png', 'roi3130.png', 'roi3138.png', 'roi3137.png', 'roi3139.png', 'roi3134.png', 'roi3135.png', 'roi3140.png','roi251.png', 'roi253.png', 'roi252.png']



from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.utils import class_weight
from keras.utils import plot_model
from sklearn import metrics


def get_images (rois):
    roi_images = []
    for i in range(len(rois)):
        roi = rois[i]
        img = cv2.imread("/Users/2020shatgiskessell/Desktop/New_Mole_Detector/ANN_Images_Fiverr/" + str(roi),0)
        try:
            img = cv2.resize(img, (8,8))
            #normalize image to 0 and 1 for sigmoud function
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #add extra dimension to match CNN input shape
            img = np.expand_dims(img, axis=2)
            roi_images.append(img)
        except Exception:
            is_mole.pop(i)
            print ("Null image")
            continue
    roi_images = np.array(roi_images)
    return roi_images

def plot_epoch_data(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_data1.png')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_data1.png')
    plt.show()

rois = []
is_mole = []

with open('/Users/2020shatgiskessell/Desktop/New_Mole_Detector/Mole_Detector_1_3/tagged_data.json') as f:
    data = json.load(f)
    rois = list(data.keys())
    is_mole = list(data.values())

roi_images = get_images (rois)
X_train, X_test, y_train, y_test = train_test_split(roi_images, is_mole, test_size = 0.2)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(8,8,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#balanace classes
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), class_weight = class_weights, epochs=30,  verbose=1)
y_pred = model.predict(X_test)
y_pred = y_pred.round()
print (y_pred)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plot_model(model, to_file='model.png')
model.save('my_mole_model_2.h5')
plot_epoch_data(history)
