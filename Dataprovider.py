import glob , os
from PIL import Image
import numpy as np
import random
import configure as cfg
from image_processing import ImageProcessing
from utils import plot_images
class DataProvider(object):
    def __init__(self):
        pass;

class WallyDataset_ver1(object):
    def __init__(self , imgdir , onehot ):
        self.imgdir = imgdir
        self.WALLY = 0
        self.NOT_WALLY = 1
        # Crawling Waldo paths
        self.waldo_paths = glob.glob(os.path.join(self.imgdir, 'waldo','*.jpg' ))
        # Waldo Validation
        self.val_waldo_paths = self.waldo_paths[-4:]
        # Waldo Train
        self.waldo_paths = self.waldo_paths[:-4]
        # Crawling not waldo paths
        self.not_waldo_paths = glob.glob(os.path.join(self.imgdir,'notwaldo' ,'*.jpg' ))
        # Not Waldo Validation
        self.val_not_waldo_paths = self.not_waldo_paths[-4:]
        # Not Waldo Train
        self.not_waldo_paths = self.not_waldo_paths[:-4]
        # Set Balance
        self.waldo_paths = self.waldo_paths *9
        # Lables
        self.labs = np.asarray([self.WALLY] * len(self.waldo_paths) + [self.NOT_WALLY] * len(self.not_waldo_paths))
        self.val_labs = np.asarray([self.WALLY] * len(self.val_waldo_paths) + [self.NOT_WALLY] * len(self.val_not_waldo_paths))
        # Validation

        if onehot :
            self.labs = self.cls2onehot(self.labs , cfg.n_classes)
            self.val_labs = self.cls2onehot(self.val_labs, cfg.n_classes)

        self.imgs = np.asarray(map(self.path2img , self.waldo_paths + self.not_waldo_paths))
        self.val_imgs = np.asarray(map(self.path2img, self.val_waldo_paths + self.val_not_waldo_paths))
        # Data Normalization
        self.imgs = self.imgs /255.
        self.val_imgs = self.val_imgs / 255.
        #
        print '# Waldo :{} \t # Not Waldo :{} \t '.format(len(self.waldo_paths) , len(self.not_waldo_paths))
        print 'Image shape : {}'.format(np.shape(self.imgs))
        print 'Labels shape : {}'.format(np.shape(self.labs))
        assert len(self.labs) == len(self.imgs) and len(self.labs) != 0
        assert not np.max(self.imgs) > 1

    def path2img(self ,path , array=True):
        img=Image.open(path).convert('RGB')
        if array :
            img=np.asarray(img)
        return img



class WallyDataset_ver2():
    def __init__(self , fg_dir , bg_dir , resize ):
        self.WALLY = 0
        self.NOT_WALLY = 1
        self.fg_paths = glob.glob(os.path.join(fg_dir , '*'))
        self.bg_paths = glob.glob(os.path.join(bg_dir, '*'))
        self.resize = resize
        print '# Foreground : {} \t # Background : {}'.format(len(self.fg_paths) , len(self.bg_paths))

        # foreground images
        image_process = ImageProcessing()
        self.fg_imgs = image_process.paths2imgs(self.fg_paths , self.resize)
        self.n_fg = len(self.fg_imgs)
        # background imaegs
        self.bg_imgs = image_process.paths2imgs(self.bg_paths[:100] , self.resize)
        self.n_bg = len(self.bg_imgs)

    def next_batch(self , fg_batchsize , bg_batchsize):
        fg_indices = random.sample(range(self.n_fg) ,fg_batchsize )
        bg_indices = random.sample(range(self.n_bg) , bg_batchsize)

        batch_fgs =self.fg_imgs[fg_indices]
        batch_bgs = self.bg_imgs[bg_indices]
        batch_xs = np.vstack([batch_fgs, batch_bgs])
        batch_ys = [self.WALLY] * fg_batchsize + [self.NOT_WALLY] * bg_batchsize

        return batch_xs ,batch_ys




if __name__ == '__main__':
    fg_dir = 'foreground/original_fg'
    bg_dir = 'background/cropped_bg'

    dataprovider = WallyDataset_ver2(fg_dir , bg_dir , resize = (64,64))
    batch_xs , batch_ys = dataprovider.next_batch(10,10)
    plot_images(batch_xs , batch_ys )










