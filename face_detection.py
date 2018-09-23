import numpy as np
import cv2
#import matplotlib library
#import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


test = cv2.imread('waldo_big.png')
# Convert RGB ==> Greys
print np.shape(test)
gray_img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
#load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('opencv/data/lbpcascades/lbpcascade_frontalface.xml')
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);

# print the number of faces found
print('Faces found: ', len(faces))

#go over list of faces and draw them as rectangles on original colored
for (x, y, w, h) in faces:
    cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 0), 2)

# write image
cv2.imwrite('find_waldo_small.png', test)