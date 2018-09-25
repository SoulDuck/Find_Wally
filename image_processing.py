#-*-coding:utf-8 -*-
import numpy as np
from PIL import Image
import cv2 , os
import glob
from utils import show_progress , get_names , plot_images , get_name
import numpy as np


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




if __name__ == '__main__':

    paths = glob.glob(os.path.join('cropped_fg','original_fg' , '*'))
    names = get_names(paths)


    img_processing = ImageProcessing()
    # Load Image from paths
    imgs = img_processing.paths2imgs(paths , None )
    # padding pad to rect
    padded_imgs = img_processing.rect2square_imgs(list(imgs))
    padded_imgs = np.asarray(padded_imgs) / 255.
    #
    imgs_coords = img_processing.generate_copped_imgs('test_imgs' ,  32, 32 ,64, 64)
    print imgs_coords.keys()
    for key in imgs_coords.keys()[:]:
        imgs, coords = imgs_coords[key]

        print np.shape(imgs)
        for i in range(50):
            plot_images(imgs[40*i:40*(i+1)] ,coords[40*i:40*(i+1)])


    # 1,33 1.png
    # 13,43 3.png
    # 3,20 3,21 2.png
























