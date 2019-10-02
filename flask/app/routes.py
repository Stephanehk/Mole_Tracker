from flask import *
from app import app
import os
from flask import Flask, render_template, request
import sys
import cv2

sys.path.append("/Users/2020shatgiskessell/Desktop/New_Mole_Detector-master/flask/app")
from models import main_blob_detector

UPLOAD_FOLDER = os.path.basename('images')
app.config['images'] = UPLOAD_FOLDER



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

file_names = []
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['images'], file.filename)
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
    file_names.append(f)
    if len(file_names) == 2:
        print ("calculating...")
        print (file_names[0])
        #run mole detection script
        x1,y1, n_moles1, img1 = main_blob_detector.main(file_names[0], 1, True)
        x2,y2, n_moles2, img2 = main_blob_detector.main(file_names[1], 2, True)
        message = "New Moles: " + str(n_moles2 - n_moles1)
        print (n_moles1)
        print (n_moles2)
        #remove images after analysing them
        os.remove(file_names[0])
        os.remove(file_names[1])
        cv2.imwrite(os.path.join("/Users/2020shatgiskessell/Desktop/New_Mole_Detector-master/flask/app/static/public/img" , 'image_a.jpg'), img1)
        cv2.imwrite(os.path.join("/Users/2020shatgiskessell/Desktop/New_Mole_Detector-master/flask/app/static/public/img" , 'image_b.jpg'), img2)
        return render_template('index.html', invalidImage=False, output_message=message, image_b = "../static/public/img/image_b.jpg", image_a =  "../static/public/img/image_a.jpg")


    #return render_template('index.html')
    return render_template('index.html', invalidImage=False, init=True)
