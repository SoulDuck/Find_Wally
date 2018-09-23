import numpy as np
import cv2
import os
#import matplotlib library
#import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#dataset_name = 'waldo_big'
dataset_name = 'waldo_original/1'
test = cv2.imread('{}.jpg'.format(dataset_name))
# Convert RGB ==> Greys
print np.shape(test)
gray_img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
#load cascade classifier training file for haarcascade
#haar_face_cascade = cv2.CascadeClassifier('opencv/data/lbpcascades/lbpcascade_frontalface.xml')


detect_list=[
    'haarcascade_eye_tree_eyeglasses.xml',
    'haarcascade_frontalcatface.xml',
    'haarcascade_frontalface_alt.xml',
    'haarcascade_lefteye_2splits.xml',
    'haarcascade_profileface.xml',
    'haarcascade_smile.xml',
    'haarcascade_eye.xml',
    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_default.xml',
    'haarcascade_licence_plate_rus_16stages.xml',
    'haarcascade_righteye_2splits.xml',
    'haarcascade_upperbody.xml',
    'haarcascade_frontalcatface_extended.xml',
    'haarcascade_frontalface_alt_tree.xml',
    'haarcascade_fullbody.xml',
    'haarcascade_lowerbody.xml',
    'haarcascade_russian_plate_number.xml',]

for detector in detect_list:
    detector_name = os.path.splitext(os.path.split(detector)[-1])
    print detector
    haar_face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/{}'.format(detector))
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);

    # print the number of faces found
    print('{} Faces found: '.format(detector_name), len(faces))

    if len(faces) >0:
        #go over list of faces and draw them as rectangles on original colored
        for (x, y, w, h) in faces:
            cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # write image
        cv2.imwrite('{}_{}.png'.format(detector_name , dataset_name), test)