#-*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
def crop_img(np_img, h_stride, w_stride, crop_h, crop_w):
    ret_coords = []
    # 마지막 꼬투리는 무시합니다
    ret_imgs = []
    h, w, ch = np.shape(np_img)
    h_hops = (h - crop_h) / h_stride + 1
    w_hops = (w - crop_w) / w_stride + 1
    print h_hops, w_hops

    for h in range(h_hops):
        for w in range(w_hops):
            h_start = h * h_stride
            h_end = ((h) * h_stride) + crop_h
            w_start = (w * w_stride)
            w_end = ((w) * w_stride) + crop_w
            ret_imgs.append(np_img[h_start: h_end, w_start: w_end, :])
            ret_coords.append([h, w])

    return ret_imgs, ret_coords


def plot_images(imgs , names=None , random_order=False , savepath=None):
    h=math.ceil(math.sqrt(len(imgs)))
    fig=plt.figure()

    for i in range(len(imgs)):
        ax=fig.add_subplot(h,h,i+1)
        if random_order:
            ind=random.randint(0,len(imgs)-1)
        else:
            ind=i
        img=imgs[ind]
        plt.imshow(img)
        if not names==None:
            ax.set_xlabel(names[ind])
    if not savepath is None:
        plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    img=Image.open('waldo_world.png').convert('RGB')
    np_img = np.asarray(img)
    imgs , coords = crop_img(np_img , 32,32,64,64)

    for img in imgs :
        plt.imshow(img)
        plt.show()



