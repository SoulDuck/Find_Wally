import numpy as np
from utils import plot_images
path ='./wally_imgs.npy'
np_imgs = np.load(path)
print len(np_imgs)
plot_images(np_imgs[:50])
plot_images(np_imgs[50:100])
plot_images(np_imgs[100:150])
plot_images(np_imgs[150:200])
plot_images(np_imgs[250:300])
plot_images(np_imgs[300:350])
plot_images(np_imgs[350:400])
plot_images(np_imgs[400:450])




