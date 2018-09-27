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
        elif locate == 'random':
            raise NotImplementedError
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

    def stride_cropping(self, np_img, h_stride, w_stride, crop_h, crop_w):
        ret_coords = []
        # 마지막 꼬투리는 무시합니다
        # 해당 스트라이트 만큼 움직이면서 이미지를 cropping 합니다
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


    def generate_cropped_imgs(self, src_dir, h_stride, w_stride, crop_h, crop_w):
        # 모든 이미지의 src dir 에 있는 모든 이미지를 crop 합니다
        # '1.jpg' : { cropped_imgs , coords } 이런 형식 으로 되어 있습니다
        # coords 형식에는 (0, 32)  0번째 행 , 32 째 열 데이터을 말하고 있습니다

        paths = glob.glob(os.path.join(src_dir, '*'))

        cropped_imgs_coords = {}
        for path in paths:
            name = get_name(path)
            img = np.asarray(Image.open(path).convert('RGB'))
            # Cropping Images
            cropped_imgs, coords = self.stride_cropping(img, h_stride, w_stride, crop_h, crop_w)
            cropped_imgs = np.asarray(cropped_imgs)
            cropped_imgs_coords[name] = [cropped_imgs , coords]

        return cropped_imgs_coords

    def divide_TVT(self , src , ratio_val , ratio_test):

        # source 을 validation , test 비율로 나눕니다
        #

        n_val = int(len(src) * ratio_val)
        n_test =  int(len(src) * ratio_test)
        src_val = src[:n_val]
        src_test = src[n_val: n_val + n_test]
        src_train= src[n_val + n_test : ]

        return src_train , src_val , src_test

    def crop_img(self, src_img , coordinate):
        # src img 을 해당 coordinate 에 맞게 cropping 합니다
        # src img 는 numpy 입니다
        # coordinate 는 x1,y1, x2 ,y2 입니다
        x1, y1, x2, y2=coordinate
        return src_img[y1:y2 ,x1 :x2]

    def cal_coord_include_target(self, src_img ,crop_size , target_coord):
        # 특정 지점(target coord )이 들어가면서 해당 넓이의 이미지들이 뽑히는것
        # target coord x1,y1,x2,y2
        # crop size (h, w )
        # src image np image
        ori_h, ori_w = np.shape(src_img)[:2]
        tx1 ,ty1 , tx2 , ty2 = target_coord
        crop_h , crop_w = crop_size
        ret_x1 = tx2 - crop_w
        ret_x2 =tx1 + crop_w
        ret_y1 =ty2 - crop_h
        ret_y2 =ty1 + crop_h
        # ret x1 이 0 보다 작으면 안됩니다 , y도 동일
        # x2 가 이미지 w 보다 크면 안됩니다 . y도 동일
        ret_x1 = max(0, ret_x1 )
        ret_x2 = min(ori_w , ret_x2 )
        ret_y1 = max(0, ret_y1 )
        ret_y2 = min(ori_h , ret_y2)

        return ret_x1 , ret_y1 , ret_x2 , ret_y2

    def guarantee_stride_cropping( self , src_img , crop_size , target_coord , stride_size ):
        # 특정 좌표 , target croodinate 가 있는 이미지가 cropping 하는 이미지에 있는 걸 보증합니다
        # 그리고 특정 stride 만큰 움직이면서 모든 이미지를 cropping 합니다
        # stride : (stride_h , stride_w)

        stride_h , stride_w= stride_size
        crop_h , crop_w = crop_size
        tx1 , ty1 , tx2 , ty2 = self.cal_coord_include_target(src_img , crop_size , target_coord)
        cropped_img = self.crop_img(src_img , coordinate= (tx1 , ty1 , tx2 , ty2))
        cropped_imgs , coords = self.stride_cropping(cropped_img , stride_h , stride_w ,crop_h , crop_w )

        return cropped_imgs , coords

    def make_tfrecord(self, tfrecord_path, resize ,*args ):
        """

        usage : (n_imgs , imgs ) ,
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
    def generate_bg(self, src_dir , h_stride, w_stride, crop_h, crop_w, save_dir ):
        """
        stride_cropping
        Usage :
            img=Image.open('waldo_world.png').convert('RGB')
        np_img = np.asarray(img)
        #imgs , coords = crop_img(np_img , 32,32,64,64)
        src_dir ='./background/original_bg'
        save_dir = './background/cropped_bg'
        generate_bg(src_dir , 64, 64, 64, 64 , save_dir)


        :return:
        """
        paths = glob.glob(os.path.join(src_dir  , '*'))
        for path in paths:
            name =os.path.splitext(os.path.split(path)[-1])[0]
            img = np.asarray(Image.open(path).convert('RGB'))
            # Cropping Image
            cropped_imgs , coords =self.stride_cropping(img,h_stride, w_stride, crop_h, crop_w)
            cropped_imgs = np.asarray(cropped_imgs)
            for i , cropped_img in enumerate(cropped_imgs ):
                Image.fromarray(cropped_img).save(os.path.join( save_dir, '{}_{}.jpg'.format(name , i)))


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
    #
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

    img_processing.make_tfrecord('train.tfrecord' , (64,64) , (n_train , fg_train_imgs) , (n_train , bg_train_imgs))
    img_processing.make_tfrecord('test.tfrecord', (64, 64), (n_test, fg_train_imgs), (n_test, bg_train_imgs))
    img_processing.make_tfrecord('val.tfrecord', (64, 64), (n_val, fg_train_imgs), (n_val, bg_train_imgs))
    #
    padded_imgs = img_processing.rect2square_imgs(list(imgs))
    padded_imgs = np.asarray(padded_imgs) / 255.
    #
    imgs_coords = img_processing.generate_cropped_imgs('test_imgs' ,  32, 32 ,64, 64)
    #
    print imgs_coords.keys()
    for key in imgs_coords.keys()[:]:
        imgs, coords = imgs_coords[key]

        np.save(key.replace('png','npy'),imgs)

        coords_indices = zip(coords , range(len(coords)))
        print np.shape(imgs)
        for i in range(50):
            plot_images(imgs[40*i:40*(i+1)] ,coords_indices[40*i:40*(i+1)])

