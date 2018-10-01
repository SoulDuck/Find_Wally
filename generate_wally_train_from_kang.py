#-*- coding:utf-8 -*-
# 강상재 쌤이 만들어 주신 데이터 셋

import os

import utils
import h5py
import numpy as np
import cv2

from skimage.transform import rescale, rotate
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
from sklearn.model_selection import train_test_split

import random
from image_processing import ImageProcessing

def load_dataset(h5_path="./prep.h5"):
    with h5py.File(h5_path) as file:
        return file['fg_data'][:], file['bg_data'][:]


def willyGenerator(fg_dataset,
                   bg_dataset,
                   crop_dim,
                   fg_ratio=0.5,
                   batch_size=64,
                   aug_funcs=[],
                   prep_funcs=[],
                   convolution=False):
    '''
    img_dim : 모델에 feeding하기 위한 input size
    batch_size : batch size

    aug_funcs : list of data augmentation func
    prep_func : list of preprocessing func
    '''
    if batch_size is None:
        # batch_size = None -> Full batch
        batch_size = fg_dataset.shape[0]

    counter = 0
    batch_image = []
    batch_label = []
    fg_idx = 0
    bg_idx = 0
    label = []
    fg_flag = True

    while True:
        counter += 1

        # picking One Image
        if np.random.random() < fg_ratio:
            fg_flag = True
            image = fg_dataset[fg_idx]
            label = np.array([0., 1.])
            if fg_idx == fg_dataset.shape[0] - 1:
                fg_idx = 0
                np.random.shuffle(fg_dataset)
            else:
                fg_idx += 1
        else:
            fg_flag = False
            image = bg_dataset[bg_idx]
            image = crop_random(image, (crop_dim[0] * 2, crop_dim[1] * 2))
            label = np.array([1., 0.])
            if bg_idx == bg_dataset.shape[0] - 1:
                bg_idx = 0
                np.random.shuffle(bg_dataset)
            else:
                bg_idx += 1

        # label format : 컨볼루션인경우, [None, Height, Width, Dim] 형태로 잡아주어야 함
        if convolution:
            label = label[np.newaxis, np.newaxis]

        # Image Normalization
        image = cv2.normalize(image, np.zeros_like(image),
                              alpha=0., beta=1.,
                              norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_32F)

        # Apply Image Augumentation
        for aug_func in aug_funcs:
            image = aug_func(image)

        # Crop Image
        if fg_flag:
            image = crop_center(image, crop_dim)  # foreground는 가운데를 crop
        else:
            image = crop_random(image, crop_dim)  # background는 랜덤에서 crop

        # Apply Image Preprocessing
        for prep_func in prep_funcs:
            image = prep_func(image)

        batch_image.append(image)
        batch_label.append(label)

        if counter == batch_size:
            yield np.stack(batch_image, axis=0), np.stack(batch_label, axis=0)
            counter = 0;
            batch_image = [];
            batch_label = []


def crop_center(image, crop_dim):
    '''
    image의 정가운데를 crop_dim에 맞추어서 잘라줌
    '''
    h, w = image.shape[:2]
    c_y, c_x = h // 2, w // 2
    out_h, out_w = crop_dim[:2]
    return image[c_y - out_h // 2:c_y + out_h - out_h // 2,
           c_x - out_w // 2: c_x + out_w - out_w // 2]


def crop_random(image, crop_dim):
    '''
    image에서 랜덤한 위치에서 crop_dim만큼 잘라 가져옴
    '''
    h, w = image.shape[:2]
    out_h, out_w = crop_dim[:2]

    crop_y = np.random.choice(range(0, h - out_h - 1))
    crop_x = np.random.choice(range(0, w - out_w - 1))

    return image[crop_y:crop_y + out_h,
           crop_x:crop_x + out_w]


# data augumentation pipeline
def rotation_func(min_angle=-8, max_angle=8):
    def func(data):
        data = np.clip(data, -1., 1.)
        rotation_angle = random.randint(min_angle, max_angle)
        return rotate(data, rotation_angle, mode='constant', cval=0.)

    return func


def rescaling_func(scale=0.1):
    def func(data):
        rescale_ratio = (1 - scale) + random.random() * scale * 2
        return rescale(data, rescale_ratio, mode='reflect')

    return func


def flip_func():
    def func(data):
        if random.random() > 0.5:
            return data[:, ::-1, :]
        else:
            return data

    return func


def random_crop_func(crop_dim=(56, 56)):
    def func(data):
        iv = random.randint(0, data.shape[0] - crop_dim[0])
        ih = random.randint(0, data.shape[1] - crop_dim[1])
        return data[iv:iv + crop_dim[0], ih:ih + crop_dim[1], :]

    return func


def random_noise_func(var=0.001):
    def func(data):
        data[:, :, :3] = random_noise(data[:, :, :3],
                                      mode='gaussian',
                                      mean=0, var=var)
        data = np.clip(data, 0.0, 1.0)
        return data

    return func


def gamma_func(min_gamma=0.9, max_gamma=1.1):
    def func(data):
        gamma = np.random.uniform(min_gamma, max_gamma)
        data[:, :, :3] = adjust_gamma(data[:, :, :3], gamma)
        data = np.clip(data, 0.0, 1.0)
        return data

    return func


def color_gamma_func(min_gamma=0.9, max_gamma=1.1):
    def func(data):
        for i in range(3):
            gamma = np.random.uniform(min_gamma, max_gamma)
            data[:, :, i] = adjust_gamma(data[:, :, i], gamma)
        data = np.clip(data, 0.0, 1.0)
        return data

    return func


def hue_func(min_gamma=0.9, max_gamma=1.1):
    def func(data):
        data = data.astype(np.float32)
        data[:, :, :3] = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2HSV)
        gamma = np.random.uniform(min_gamma, max_gamma)
        data[:, :, 0] = adjust_gamma(data[:, :, 0], gamma)
        data[:, :, :3] = cv2.cvtColor(data[:, :, :3], cv2.COLOR_HSV2RGB)
        data = np.clip(data, 0.0, 1.0)
        return data

    return func


def saturation_func(min_gamma=0.9, max_gamma=1.1):
    def func(data):
        data = data.astype(np.float32)
        data[:, :, :3] = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2HSV)
        gamma = np.random.uniform(min_gamma, max_gamma)
        data[:, :, 1] = adjust_gamma(data[:, :, 1], gamma)
        data[:, :, :3] = cv2.cvtColor(data[:, :, :3], cv2.COLOR_HSV2RGB)
        data = np.clip(data, 0.0, 1.0)
        return data

    return func


if __name__ == '__main__':


    # load data generator
    fg, bg = load_dataset()
    # data augumentation list
    aug_funcs = [rotation_func(min_angle=-8,max_angle=8),
                 rescaling_func(scale=0.2),
                 flip_func(),
                 random_crop_func(crop_dim=(51,51)),
                 random_noise_func(),
                 gamma_func(),
                 color_gamma_func(),
                 hue_func(max_gamma=1.3),
                 saturation_func(min_gamma=0.9,max_gamma=1.3)]

    valid_funcs = [rotation_func(min_angle=-8,max_angle=8),
                   rescaling_func(scale=0.2),
                   flip_func(),
                   random_crop_func(crop_dim=(51,51))]

    train_fg, valid_fg = train_test_split(fg,test_size=0.2)
    train_bg, valid_bg = train_test_split(bg,test_size=0.2)

    valid_generator = willyGenerator(valid_fg, valid_bg,
                                     crop_dim=(48,48,3),
                                     fg_ratio=0.5,
                                     batch_size=1000,
                                     aug_funcs=valid_funcs,
                                     convolution=True)

    train_generator = willyGenerator(train_fg, train_bg,
                                     crop_dim=(48, 48, 3),
                                     fg_ratio=0.5,
                                     batch_size=60,
                                     aug_funcs=aug_funcs,
                                     convolution=True)

    valid_x, valid_y = next(valid_generator)

    # EveryNN 에 넣기 위해서..
    train_xs, train_ys = next(train_generator)

    trainXs_list = []
    trainYs_list = []

    utils.plot_images(train_xs , train_ys )
    max_iter = 2
    for i in range(max_iter):
        utils.show_progress(i , max_iter)
        train_xs, train_ys = next(train_generator)
        trainXs_list.append(train_xs)
        trainYs_list.append(train_ys)

    trainYs_list = np.squeeze(trainYs_list)
    np.save('train_imgs.npy' , train_xs)
    np.save('train_labs.npy' , train_ys)




    train_cls = np.argmax(train_ys , axis=1)
    notwally_indices = np.where([train_cls == 0])[1]
    wally_indices =  np.where([train_cls == 1])[1]

    wally_train_xs = train_xs[wally_indices]
    notwally_train_xs = train_xs[notwally_indices]

    assert len(wally_train_xs) + len(notwally_train_xs) == len(train_xs)

    img_prc=ImageProcessing()
    img_prc.make_tfrecord('wally.tfrecord', (48,48), (len(wally_train_xs), wally_train_xs),
                          (len(notwally_train_xs), notwally_train_xs))



