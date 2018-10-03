import numpy as np
from utils import plot_images
path ='./wally_imgs.npy'
np_imgs = np.load(path)
print len(np_imgs)
for i in range(12):
    plot_images(np_imgs[10*i:10*(i+1)])


