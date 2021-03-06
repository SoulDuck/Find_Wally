#-*- coding:utf-8 -*-
# 강상재 쌤이 만들어 주신 데이터 셋 , 라즈베리 파이로 찍은 사진들
import copy
import os , glob
import h5py
import numpy as np
import cv2
import utils
import pandas as pd
from PIL import Image
import random
from image_processing import ImageProcessing
#import matplotlib.pyplot as plt
# 얼굴이 있는 사진을 추출합니다
def extract_wallybody(dirpath , anns_path):
    img_prc =ImageProcessing()
    anns = open(anns_path, 'r')

    paths = glob.glob('{}/*.jpg'.format(dirpath))
    names = os.listdir(dirpath)

    # Body 가 들어있는 이미지의 정보를 가져옵니다
    lines = anns.readlines()
    body_dict = {}
    for line in lines[1:] :
        fpath, x1, x2, y1, y2 = line.split(',')
        x1, x2, y1, y2 = map(lambda ele: int(ele.strip()), [x1, x2, y1, y2])
        name = utils.get_name(fpath)
        # first
        if not name in body_dict.keys():
            body_dict[name] = [(x1, x2, y1, y2 )]
        else:
            body_dict[name].append((x1, x2, y1, y2))

    fg_imgs_list = []
    bg_imgs_list = []
    fgs=[]
    # get wally face list
    for p,path in enumerate(paths):
        name = utils.get_name(path)
        img = np.asarray(Image.open(path).convert('RGB'))
        # extract wally
        if name in body_dict.keys():
            for i,coord in enumerate(body_dict[name]):
                x1, x2, y1, y2 = coord
                fg = img[y1:y2 , x1:x2 , : ]

                fgs.append(fg)
                fg_imgs , fg_coords = img_prc.guarantee_stride_cropping(img , (400,400) , [x1,y1,x2,y2] , (25,25))
                if len(fg_imgs) ==0:
                    print path , x2-x1, y2-y1
                else:
                    fg_imgs = img_prc.resize_npImages(fg_imgs, (80, 80))
                    fg_imgs_list.append(fg_imgs)


                img=copy.deepcopy(img)
                # fill rectangle for extract back grounds images
                cv2.rectangle(img,  (x1,y1) , (x2,y2) , (0,0,0), -1)

        bg_imgs , bg_coords = img_prc.stride_cropping(img, 200 ,200 , 400 ,400)
        bg_imgs= img_prc.resize_npImages(bg_imgs, (80, 80))
        bg_imgs_list.append(bg_imgs)

    fgs = np.vstack(fg_imgs_list)
    bgs = np.vstack(bg_imgs_list)

    return fgs , bgs


if __name__ == '__main__':
    sec_dir = './wally_dataset/second_dataset'
    thr_dir = './wally_dataset/third_dataset'
    sec_anns = './wally_dataset/second_dataset/body_crop.csv'
    thr_anns = './wally_dataset/third_dataset/body_crop.csv'

    #
    sec_fgs ,sec_bgs =extract_wallybody(sec_dir , sec_anns )



    indices =range(len(sec_bgs))
    random.shuffle(indices)

    thr_fgs, thr_bgs = extract_wallybody(thr_dir, thr_anns)



    fgs = np.vstack([sec_fgs , thr_fgs])
    bgs = np.vstack([sec_bgs, sec_fgs])

    np.save('wally_dataset/wally_imgs.npy' , fgs)
    np.save('wally_dataset/background_imgs.npy' , bgs)

    img_prc = ImageProcessing()
    train_fg , test_fg , val_fg =img_prc.divide_TVT(fgs, 0.1 , 0.1 , True )
    train_bg, test_bg, val_bg = img_prc.divide_TVT(bgs, 0.1, 0.1 , True )

    np.save('wally_dataset/wally_train_imgs.npy', train_fg)
    np.save('wally_dataset/wally_test_imgs.npy', test_fg)
    np.save('wally_dataset/wally_val_imgs.npy', val_fg)

    np.save('wally_dataset/background_train_imgs.npy', train_bg)
    np.save('wally_dataset/background_test_imgs.npy', test_bg)
    np.save('wally_dataset/background_val_imgs.npy', val_bg)





