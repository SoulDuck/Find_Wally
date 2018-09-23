import numpy as np
import cv2
#import matplotlib library
#import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


test = cv2.imread('test.png')
# Convert RGB ==> Greys
gray_img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
#load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);

# print the number of faces found
print('Faces found: ', len(faces))
