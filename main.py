import tensorflow as tf
import numpy as np
import configure as cfg
from models import Models
from Dataprovider import Dataprovider
from utils import show_progress
import matplotlib.pyplot as plt
if __name__ == '__main__':
    dp = Dataprovider('./Hey-Waldo/128' , True)
    models = Models(2 , (128,128,3))
    batch_xs , batch_ys = dp.next_batch(batch_size=cfg.batch_size )
    for i,img in enumerate(batch_xs):
        plt.imshow(img)
        print batch_ys[i]
        plt.show()


    for step in range(cfg.max_iter):
        show_progress(step , cfg.max_iter)
        models.training(batch_xs , batch_ys , cfg.lr)
        if step % cfg.ckpt  == 0 :
            print 'Validation ... '
            pred, pred_cls, cost, accuracy =models.eval(dp.val_imgs , dp.val_labs,)
            models.save_models('models/{}.ckpt'.format(step))
            print accuracy
















