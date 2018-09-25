#-*-coding:utf-8 -*-
import numpy as np
from PIL import Image
import cv2 , os
import glob
from utils import show_progress , get_names , plot_images , get_name
import numpy as np
import random
import tensorflow as tf
import sys

class ImageProcessing(object):
    def __init__(self):
        pass;

    def get_square(self , np_img):
        assert np.ndim(np_img) ==3 , 'Image Dimension has to be 3 , if you have grey image , please reshape to 3 '
        h, w, ch = np.shape(np_img)
        max_value = np.max([h, w])
        return np.zeros([max_value, max_value ,ch])

    def paddding(self, np_fg , np_bg , locate = 'center'):
        """
        1.np_bg 에다가 np_fg을 덮어 씌웁니다

        :param np_fg:
        :param np_bg:
        :return:
        """

        assert np.ndim(np_fg) == 3 and np.ndim(np_bg) == 3
        if locate == 'center':
            fg_h, fg_w, fg_ch =np.shape(np_fg)
            bg_h, bg_w, bg_ch = np.shape(np_bg)

            bg_ctr_x, bg_ctr_y = int(bg_w / 2), int(bg_h / 2)
            fg_ctr_x, fg_ctr_y = int(fg_w / 2), int(fg_h / 2)

            start_x =bg_ctr_x - fg_ctr_x
            end_x = bg_ctr_x + fg_ctr_x

            start_y = bg_ctr_y - fg_ctr_y
            end_y = bg_ctr_y + fg_ctr_y

            np_bg[start_y: end_y, start_x: end_x] = np_fg
        else:
            raise NotImplementedError
        return np_bg

    def rect_2_square(self , img):
        # Get Square Background
        np_bg = self.get_square(img)
        # Padding
        return self.paddding(img , np_bg , 'center')

    def rect2square_imgs(self , imgs):

        print np.shape(imgs)
        ret_imgs = []
        count = 0
        for img in imgs:
            show_progress(count,len(imgs))
            padded_img = self.rect_2_square(img)
            ret_imgs.append(padded_img)
            count += 1
        return ret_imgs
    def path2img(self , path , resize):

        img = Image.open(path).convert('RGB')
        if not resize is None:
            img = img.resize(resize)
        return np.asarray(img)

    def paths2imgs(self , paths , resize):
        np_imgs = []
        for path in paths:
            np_img=self.path2img(path , resize)
            np_imgs.append(np_img)
        return np.asarray(np_imgs)

    def crop_img(self, np_img, h_stride, w_stride, crop_h, crop_w):
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


    def generate_copped_imgs(self, src_dir, h_stride, w_stride, crop_h, crop_w):
        paths = glob.glob(os.path.join(src_dir, '*'))

        cropped_imgs_coords = {}

        for path in paths:
            name = get_name(path)
            img = np.asarray(Image.open(path).convert('RGB'))
            # Cropping Images
            cropped_imgs, coords = self.crop_img(img, h_stride, w_stride, crop_h, crop_w)
            cropped_imgs = np.asarray(cropped_imgs)
            cropped_imgs_coords[name] = [cropped_imgs , coords]

        return cropped_imgs_coords

    def divide_TVT(self , src , ratio_val , ratio_test):

        """divide source into train ,validation ,test """

        n_val = int(len(src) * ratio_val)
        n_test =  int(len(src) * ratio_test)
        src_val = src[:n_val]
        src_test = src[n_val: n_val + n_test]
        src_train= src[n_val + n_test : ]

        return src_train , src_val , src_test




def make_tfrecord(tfrecord_path, resize ,*args ):
    """
    img source 에는 두가지 형태로 존재합니다 . str type 의 path 와
    numpy 형태의 list 입니다.
    :param tfrecord_path: e.g) './tmp.tfrecord'
    :param img_sources: e.g)[./pic1.png , ./pic2.png] or list flatted_imgs
    img_sources could be string , or numpy
    :param labels: 3.g) [1,1,1,1,1,0,0,0,0]
    :return:
    """
    if os.path.exists(tfrecord_path):
        print tfrecord_path + 'is exists'
        return
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    flag=True
    n_total =0
    counts = []
    for i,arg in enumerate(args):
        print 'Label :{} , # : {} '.format(i , arg[0])
        n_total += arg[0]
        counts.append(0)

    while(flag):
        label=random.randint(0,len(args)-1)
        n_max = args[label][0]
        if counts[label] < n_max:
            imgs = args[label][1]
            n_imgs = len(args[label][1])
            ind = counts[label] % n_imgs
            np_img = imgs[ind]
            counts[label] += 1
        elif np.sum(np.asarray(counts)) ==  n_total:
            for i, count in enumerate(counts):
                print 'Label : {} , # : {} '.format(i, count )
            flag = False
        else:
            continue;

        height, width = np.shape(np_img)[:2]

        msg = '\r-Progress : {0}'.format(str(np.sum(np.asarray(counts))) + '/' + str(n_total))
        sys.stdout.write(msg)
        sys.stdout.flush()
        if not resize is None:
            np_img = np.asarray(Image.fromarray(np_img).resize(resize, Image.ANTIALIAS))
        raw_img = np_img.tostring()  # ** Image to String **
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'raw_image': _bytes_feature(raw_img),
            'label': _int64_feature(label),
            'filename': _bytes_feature(tf.compat.as_bytes(str(ind)))
        }))
        writer.write(example.SerializeToString())
    writer.close()



if __name__ == '__main__':

    paths = glob.glob(os.path.join('foreground','original_fg' , '*'))
    names = get_names(paths)

    # Foreground
    img_processing = ImageProcessing()
    # Load Image from paths
    imgs = img_processing.paths2imgs(paths , (64,64))
    print 'foreground images : {}'.format(np.shape(imgs))
    #
    fg_train_imgs , fg_val_imgs, fg_test_imgs= img_processing.divide_TVT(imgs , 0.1 ,0.1)
    np.save('fg_train.npy' , fg_train_imgs)
    np.save('fg_test.npy', fg_test_imgs)
    np.save('fg_val.npy', fg_val_imgs)


    # Background
    paths = np.asarray(glob.glob(os.path.join('background', 'cropped_bg', '*')))
    names = np.asarray(get_names(paths))
    indices = random.sample(range(len(paths)) , len(paths))[:7400]
    paths = paths[indices]
    names = names[indices]
    imgs = img_processing.paths2imgs(paths, (64, 64))
    print 'background images : {}'.format(np.shape(imgs))

    bg_train_imgs, bg_val_imgs, bg_test_imgs = img_processing.divide_TVT(imgs, 0.1, 0.1)
    np.save('bg_train.npy', bg_train_imgs)
    np.save('bg_test.npy', bg_test_imgs)
    np.save('bg_val.npy', bg_val_imgs)


    #Make Tensorflow tfrecords
    """
    usage :    make_tfrecord(train_tfrecord_path, None, (len(label_0_train), label_0_train), (len(label_0_train), label_1_train),
                  (len(label_0_train), label_2_train) )
    """

    n_train = len(bg_train_imgs)
    n_test = len(bg_test_imgs)
    n_val = len(bg_val_imgs)
    make_tfrecord('train.tfrecord' , (64,64) , (n_train , fg_train_imgs) , (n_train , bg_train_imgs))
    make_tfrecord('test.tfrecord', (64, 64), (n_test, fg_train_imgs), (n_test, bg_train_imgs))
    make_tfrecord('val.tfrecord', (64, 64), (n_val, fg_train_imgs), (n_val, bg_train_imgs))
    #

    padded_imgs = img_processing.rect2square_imgs(list(imgs))
    padded_imgs = np.asarray(padded_imgs) / 255.
    #
    imgs_coords = img_processing.generate_copped_imgs('test_imgs' ,  32, 32 ,64, 64)

    print imgs_coords.keys()
    for key in imgs_coords.keys()[:]:
        imgs, coords = imgs_coords[key]

        np.save(key.replace('png','npy'),imgs)

        coords_indices = zip(coords , range(len(coords)))
        print np.shape(imgs)
        for i in range(50):

            plot_images(imgs[40*i:40*(i+1)] ,coords_indices[40*i:40*(i+1)])


    # [1,33](73) ,[1,32](72) 1.png
    # [13,43](875) [13,42](874) 3.png
    # [3,20](146) [3,21](147)    2.png
























