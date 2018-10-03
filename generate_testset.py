#-*- coding:utf-8 -*-


"""
2018.10.2 일날 rasp 로 찍은 월리 사진을 데이터 셋에 넣음

"""
import glob , os
from image_processing import ImageProcessing
import numpy as np
from PIL import Image
import utils

root_root_dir = '/mnt/Find_Wally/wally_dataset'
second_dir=os.path.join( root_root_dir ,'/second_dataset')
thrid_dir = os.path.join(root_root_dir , 'third_dataset')
root_save_dir = 'wally_raspCam_np'


img_prc = ImageProcessing()
sec_paths = glob.glob(os.path.join(second_dir , '*.jpg'))
trd_paths = glob.glob(os.path.join(thrid_dir , '*.jpg'))

assert len(sec_paths) != 0 and len(trd_paths) != 0


tmp_dict = {'second' : sec_paths , 'thrid' : trd_paths}
for key in tmp_dict:
    paths = tmp_dict[key]
    save_dir = os.path.join(root_save_dir, 'second')
    utils.makedir(save_dir)
    for path in paths :
        name = utils.get_name(path)
        img = np.asarray(Image.open(path).convert('RGB'))
        # Cropping
        imgs = img_prc.stride_cropping(img , 200 , 200 , 400 ,400)
        save_path = os.path.join(save_dir,name.replace('jpg', 'npy'))
        np.save(save_path , imgs )

