import glob , os
from PIL import Image
import numpy as np
import random
import configure as cfg
class Dataprovider(object):
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
        self.waldo_paths = self.waldo_paths * 9
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

    def next_batch(self , batch_size):

        indices = random.sample(range(len(self.labs)) , batch_size)
        batch_ys = self.labs[indices]
        batch_xs = self.imgs[indices]
        return batch_xs , batch_ys
    def cls2onehot(self, cls , depth):
        cls=cls.astype(np.int32)
        labels = np.zeros([len(cls), depth] , dtype=np.int32)
        for i, ind in enumerate(cls):
            labels[i][ind:ind + 1] = 1
        return labels


if __name__ == '__main__':
    dataprovider = Dataprovider('./Hey-Waldo/256')







