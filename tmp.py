import numpy as np
from utils import plot_images
path ='./wally_imgs.npy'
np_imgs = np.load(path)
print len(np_imgs)
plot_images(np_imgs)


