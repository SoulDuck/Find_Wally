#-*- coding:utf-8 -*-
import glob , os
from PIL import Image
import numpy as np
import random
import configure as cfg
from image_processing import ImageProcessing
from utils import plot_images  , cls2onehot , get_names
import matplotlib.pyplot as plt
import cv2
class DataProvider(object):
    def __init__(self):
        pass;



class Wally_dataset(object):
    def __init__(self):
        self.img_prc=ImageProcessing
    def get_background_imgs(self  , n_bg  , savedir ):
        bg_train_path = os.path.join(savedir, 'bg_train.npy')
        bg_test_path = os.path.join(savedir, 'bg_test.npy')
        bg_val_path = os.path.join(savedir, 'bg_val.npy')
        paths = np.asarray(glob.glob(os.path.join('background', 'cropped_bg', '*')))
        names = np.asarray(get_names(paths))

        if os.path.exists(bg_train_path) and os.path.exists(bg_test_path) and os.path.exists(bg_val_path):
            self.bg_train_imgs=np.load(bg_train_path)
            self.bg_test_imgs=np.load(bg_test_path)
            self.bg_val_imgs=np.load(bg_val_path)
        else:
            print 'Generating Not Wally Data....'
            indices = random.sample(range(len(paths)), len(paths))[:n_bg]
            #
            paths = paths[indices]
            #
            imgs = self.img_prc.paths2imgs(paths, (64, 64))
            print 'background images : {}'.format(np.shape(imgs))

            # get background train ,val test images
            self.bg_train_imgs, self.bg_val_imgs, self.bg_test_imgs = self.img_prc.divide_TVT(imgs, 0.1, 0.1)
            # Save Numpy
            np.save(os.path.join(savedir,'bg_train.npy'), self.bg_train_imgs)
            np.save(os.path.join(savedir,'bg_test.npy'), self.bg_test_imgs)
            np.save(os.path.join(savedir,'bg_val.npy'), self.bg_val_imgs)



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
    """
    Usage :
    fg_dir = 'foreground/original_fg'
    bg_dir = 'background/cropped_bg'
    dataprovider = WallyDataset_ver2(fg_dir , bg_dir , resize = (64,64))
    batch_xs , batch_ys = dataprovider.next_batch(10,10)
    plot_images(batch_xs , batch_ys )

    """
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
        self.bg_imgs = image_process.paths2imgs(self.bg_paths[:] , self.resize)
        self.n_bg = len(self.bg_imgs)

    def next_batch(self , fg_batchsize , bg_batchsize , normalization):
        fg_indices = random.sample(range(self.n_fg) ,fg_batchsize )
        bg_indices = random.sample(range(self.n_bg) , bg_batchsize)

        batch_fgs =self.fg_imgs[fg_indices]
        batch_bgs = self.bg_imgs[bg_indices]

        batch_xs = np.vstack([batch_fgs, batch_bgs])
        batch_ys = [self.WALLY] * fg_batchsize + [self.NOT_WALLY] * bg_batchsize
        batch_ys = cls2onehot(batch_ys ,depth = 2 )

        indices = random.sample(range(len(batch_ys)) , len(batch_ys))
        batch_xs = batch_xs[indices]
        batch_ys = batch_ys[indices]

        if normalization:
            batch_xs = batch_xs/255.
        return batch_xs ,batch_ys

class WallyDataset_ver3():
    def __init__(self ,imgdir ,anns ):
        self.imgdir = imgdir
        self.anns = anns

        self.paths = glob.glob(os.path.join(self.imgdir , '*.jpg'))
        self.names = get_names(self.paths)
        # read csv
        f = open(self.anns,'r')
        lines = f.readlines()
        # not include first line
        ctr_xs = []
        ctr_ys = []
        fnames = []
        for line in lines[1:]:
            ctr_x, ctr_y, filename =line.split(',')
            ctr_xs.append(int(ctr_x))
            ctr_ys.append(int(ctr_y))
            fnames.append(filename.strip())
        self.img_infos = zip(fnames , ctr_xs ,ctr_ys)

    def show_wally(self):
        for name , x ,y  in self.img_infos:
            path = os.path.join(self.imgdir, name)
            img = np.asarray(Image.open(path).convert('RGB'))
            cv2.rectangle(img, (x-50,y-50),(x+50,y+50),(255,0,0),2)
            plt.imshow(img)
            plt.show()

class WallyDataset_ver4(Wally_dataset):
    """
    Class Usage :

    1. self.infos 는
        {이미지이름.jpg :
        {'whole_img' : 이름 원본 이미지와(Numpy)} ,
        {'face_img'  , 원본 이미지 (Numpy)}
        {'coords': face coord }
        형태로 저장되어 있습니다.

    2. 월리 얼굴 형태를 포함해서 cropping 합니다
        def cropping_with_face:
        src_img : 원본 이미지입니다  , self.infos['whole_img] 에서 가져와 사용하면 됩니다.
        crop_size : 얼마만큼 crop 할지 결정합니다
        coord : 얼굴 좌표를 말합니다  ,self.infos['coords'] 에서 가져와 사용하면 됩니다.
        stride_size
    """
    def __init__(self , wholeImg_dir , faceImg_dir  , anns ):
        self.img_prc = ImageProcessing()
        self.wholeImg_dir = wholeImg_dir
        self.faceImg_dir = faceImg_dir
        self.anns = anns
        #
        self.face_paths = glob.glob(os.path.join(faceImg_dir,'*.jpg'))
        self.whole_paths = glob.glob(os.path.join(wholeImg_dir, '*.jpg'))
        self.names = get_names(self.face_paths)
        #
        f = open(self.anns,'r')
        lines = f.readlines()
        # not include first line
        self.infos= {}
        #
        print '# line {}'.format(len(lines))

        for line in lines[1:][:]:
            fname,x1,x2,y1,y2,w,h = line.split(',')
            x1, x2, y1, y2, w, h = map(lambda x : int(x.strip()) , [x1, x2, y1, y2, w, h])
            coord = [ x1, y1, x2, y2, w, h ]

            whole_img = self.img_prc.path2img(os.path.join(self.wholeImg_dir , fname) , resize = None )
            face_img = self.img_prc.path2img(os.path.join(self.faceImg_dir, fname) , resize = None)
            self.infos[fname] = {'coord' : coord , 'whole_img' : whole_img , 'face_img':face_img }

    def cropping_with_face(self,src_img , crop_size , coord , stride_size):
        # src_img = 원본 이미지
        # crop = (crop_h , crop_w)
        # coord = [x1,y1,x2,y2]
        # stride_size = stride_h , stride_w
        cropped_imgs , coords = self.img_prc.guarantee_stride_cropping(src_img, crop_size ,coord,  stride_size)
        return cropped_imgs


    def generate_tfrecord(self , tfrecord_path , n_fg ,fg_imgs , n_bg , bg_imgs):
        self.img_prc.make_tfrecord(tfrecord_path ,None ,(n_fg , fg_imgs) , (n_bg , bg_imgs))


    def get_wallyface(self):
        fg_train_savepath = os.path.join('Wally_ver3', 'numpy_imgs', 'fg_train.npy')
        fg_test_savepath = os.path.join('Wally_ver3', 'numpy_imgs', 'fg_test.npy')
        fg_val_savepath = os.path.join('Wally_ver3', 'numpy_imgs', 'fg_val.npy')

        if os.path.exists(fg_train_savepath) and os.path.exists(fg_test_savepath) and os.path.exists(fg_val_savepath):
            self.fg_train_imgs=np.load(fg_train_savepath)
            self.fg_test_imgs = np.load(fg_test_savepath)
            self.fg_val_imgs = np.load(fg_val_savepath)
        else:
            print 'Generating WallyFace Data....'
            fg_list = []
            for key in wally_dp.infos.keys():
                target_coords = self.infos[key]['coord']
                x1, y1, x2, y2 = target_coords[:4]
                whole_img = self.infos[key]['whole_img']
                # Wally 얼굴을 포함하는걸 보장하며 crop 합니다
                cropped_imgs = self.cropping_with_face(whole_img, (64, 64), [x1, y1, x2, y2], (1, 1))
                # Save crop
                np.save('./Wally_ver3/cropped_img_with_face/{}'.format(os.path.splitext(key)[0]), arr=cropped_imgs)
                fg_imgs = np.load('./Wally_ver3/cropped_img_with_face/{}.npy'.format(os.path.splitext(key)[0]))
                fg_list.append(fg_imgs)
            fg_imgs = np.vstack(fg_list)
            print 'foreground shape : {}'.format(np.shape(fg_imgs))

            # divide Train , Val , Test
            self.fg_test_imgs = fg_imgs[:5000]
            self.fg_val_imgs = fg_imgs[5000:5000 * 2]
            self.fg_train_imgs = fg_imgs[5000 * 2:]

            # save imgs to numpy
            np.save(fg_train_savepath ,self.fg_train_imgs)
            np.save(fg_test_savepath ,self.fg_test_imgs)
            np.save( fg_val_savepath ,self.fg_val_imgs)
    def get_train(self):
        imgs = np.vstack([self.fg_train_imgs ,self.bg_train_imgs ])
        labs = [0]*len(self.fg_train_imgs) + [1]*len(self.bg_train_imgs)
        train_savepath = os.path.join('Wally_ver3', 'numpy_imgs', 'train.npy')
        np.save(train_savepath , self.fg_train_imgs)




if __name__ == '__main__':
    face_imgdir = './Wally_ver3/face_images'
    whole_imgdir = './Wally_ver3/whole_images'
    anns = './Wally_ver3/face_mask.csv'
    #
    wally_dp = WallyDataset_ver4(whole_imgdir, face_imgdir, anns)
    # Get background images
    wally_dp.get_background_imgs(50000 , savedir='Wally_ver3/numpy_imgs')
    # Get foreground images
    wally_dp.get_wallyface()



    print 'Wally train : {} \t validation : {}\t test : {}'.format(wally_dp.fg_train_imgs.shape,
                                                                            wally_dp.fg_val_imgs.shape,
                                                                            wally_dp.fg_test_imgs.shape)

    print 'Not Wally train : {} \t validation : {}\t test : {}'.format(wally_dp.bg_train_imgs.shape,
                                                                                wally_dp.bg_val_imgs.shape,
                                                                                wally_dp.bg_test_imgs.shape)

    # tfrecord files train
    wally_dp.generate_tfrecord('Wally_ver3/tfrecords/train.tfrecord', len(wally_dp.fg_train_imgs),
                               wally_dp.fg_train_imgs, len(wally_dp.bg_train_imgs), wally_dp.bg_train_imgs)
    # tfrecord files test
    wally_dp.generate_tfrecord('Wally_ver3/tfrecords/test.tfrecord', len(wally_dp.fg_test_imgs),
                               wally_dp.fg_test_imgs, len(wally_dp.bg_test_imgs), wally_dp.bg_test_imgs)
    # tfrecord files val
    wally_dp.generate_tfrecord('Wally_ver3/tfrecords/val.tfrecord', len(wally_dp.fg_val_imgs),
                               wally_dp.fg_val_imgs, len(wally_dp.bg_val_imgs), wally_dp.bg_val_imgs)

