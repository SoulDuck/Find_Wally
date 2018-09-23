import numpy as np
import cv2
import os
import configure as cfg
#import matplotlib library
#import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#dataset_name = 'waldo_big'
dataset_name = 'waldo_world'
test = cv2.imread('{}.png'.format(dataset_name))
# Convert RGB ==> Greys
print np.shape(test)
gray_img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
#load cascade classifier training file for haarcascade
#haar_face_cascade = cv2.CascadeClassifier('opencv/data/lbpcascades/lbpcascade_frontalface.xml')

detect_list = cfg.facedetect_list
for detector in detect_list:
    detector_name = os.path.splitext(os.path.split(detector)[-1])
    print detector
    haar_face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/{}'.format(detector))

    scale_factors = list(np.asarray(range(0, 50)) * 0.1)
    for scale_factor in scale_factors:
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=5);
        # print the number of faces found
        print('{} Faces found: '.format(detector_name), len(faces))
        if len(faces) >0:
            #go over list of faces and draw them as rectangles on original colored
            for (x, y, w, h) in faces:
                cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # write image
                cv2.imwrite('{}_{}_{}.png'.format(scale_factor, detector_name, dataset_name), test)