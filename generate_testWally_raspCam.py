from image_processing import ImageProcessing
import numpy as np
import os
import utils
img_prc = ImageProcessing()
wally_testdir = './wally_raspCam'
imgs_coords = img_prc.generate_cropped_imgs(wally_testdir , 24,24,48,48)


for key in imgs_coords:
    imgs , coord =imgs_coords[key]
    np.save(file = os.path.join(wally_testdir , key.replace('.jpg' , '.npy')), arr = imgs )








