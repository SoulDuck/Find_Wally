import tensorflow as tf
import numpy as np
import configure as cfg
from models import Models
from Dataprovider import WallyDataset_ver2
from utils import show_progress
if __name__ == '__main__':
    fg_dir = 'cropped_fg/original_fg'
    bg_dir = 'background/cropped_bg'
    # dp == DataProvider
    dp = WallyDataset_ver2(fg_dir, bg_dir, resize=(64, 64))


    models = Models(2 , (256,256,3))
    batch_xs , batch_ys = dp.next_batch(fg_batchsize=30 , bg_batchsize=30 )


    for step in range(cfg.max_iter):
        show_progress(step , cfg.max_iter)
        train_cost = models.training(batch_xs , batch_ys , cfg.lr)
        if step % cfg.ckpt  == 0 :
            print 'Validation ... '
            pred, pred_cls, eval_cost, accuracy =models.eval(dp.val_imgs , dp.val_labs,)
            models.save_models('models/{}.ckpt'.format(step))
            print accuracy
            print eval_cost
            print train_cost


