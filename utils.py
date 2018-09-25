#-*- coding:utf-8 -*-
import sys
import os, glob
import math
import matplotlib.pyplot as plt
import random

def show_progress(i,max_iter):
    msg='\r progress {}/{}'.format(i, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()

def rename(srcdir , reg):
    # '이름을 숫자로 다 바꾸 버립니다 '
    paths = glob.glob(os.path.join(srcdir ,reg))
    print '# {}'.format(reg)

    for i, path in enumerate(paths):
        root_dir, name = os.path.split(path)
        name = os.path.splitext(root_dir)[0]
        os.rename(path, os.path.join(root_dir, str(i) + '.jpg'))

def get_name(path):
    return os.path.split(path)[-1]
def get_names(paths):
    return map(get_name , paths)



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
    rename('cropped_fg/original_fg' , '*')
