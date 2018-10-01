from utils import plot_images
import numpy as np
from utils import plot_images
path ='/Users/seongjungkim/PycharmProjects/Find_Wally/test_imgs/fg_test_imgs.npy'
np_imgs = np.load(path)
plot_images(np_imgs)



