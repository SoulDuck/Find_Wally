from image_processing import ImageProcessing
import numpy as np
import utils
img_prc = ImageProcessing()
imgs_coords = img_prc.generate_cropped_imgs('/Users/seongjungkim/PycharmProjects/Find_Wally/wally_raspCam' , 24,24,48,48)


for key in imgs_coords:
    imgs , coord =imgs_coords[key]
    print np.shape(imgs)
    utils.plot_images(imgs[:100])
    exit()





