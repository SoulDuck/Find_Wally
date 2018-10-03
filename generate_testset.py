#-*- coding:utf-8 -*-


"""
2018.10.2 일날 rasp 로 찍은 월리 사진
데이터 셋 만드는 법


 
"""
second_dir='/Users/seongjungkim/Desktop/wally_dataset/second_dataset'
thrid_dir = '/Users/seongjungkim/Desktop/wally_dataset/third_dataset'
root_save_dir = 'wally_raspCam_np'
from image_processing import ImageProcessing
import glob , os
import numpy as np
from PIL import Image
import utils
img_prc = ImageProcessing()
img_prc.stride_cropping()
sec_paths = glob.glob(os.path.join(second_dir , '*.jpg'))
trd_paths = glob.glob(os.path.join(thrid_dir , '*.jpg'))



tmp_dict = {'second' : sec_paths , 'thrid' : trd_paths}
for key in tmp_dict:
    paths = tmp_dict[key]
    save_dir = os.path.join(root_save_dir, 'second')

    for path in paths :
        name = utils.get_name(path)
        img = np.asarray(Image.open(path).convert('RGB'))
        np_imgs = img_prc.stride_cropping(img , 200 ,200 , 400 ,400)
        save_path = os.path.join(save_dir,name.replace('jpg', 'npy'))
        np.save(save_path , np_imgs )

