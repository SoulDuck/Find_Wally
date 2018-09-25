import tensorflow as tf
import numpy as np
import configure as cfg
from models import Models
from Dataprovider import WallyDataset_ver2
from utils import show_progress
from eval import Eval
if __name__ == '__main__':
    fg_dir = 'foreground/original_fg'
    bg_dir = 'background/cropped_bg'
    # dp == DataProvider
    dp = WallyDataset_ver2(fg_dir, bg_dir, resize=(64, 64))
    # Setting models
    models = Models(n_classes = 2 , img_shape = (64,64,3))
    # Get batch xs , ys
    batch_xs , batch_ys = dp.next_batch(fg_batchsize=30 , bg_batchsize=30 )
    # Training
    eval=Eval()
    for step in range(cfg.max_iter):
        show_progress(step , cfg.max_iter)
        train_cost = models.training(batch_xs , batch_ys , cfg.lr)
        if step % cfg.ckpt  == 0 :
            print 'Validation ... '
            #pred_op, pred_cls, eval_cost, accuracy =models.eval(dp.val_imgs , dp.val_labs,)
            acc = eval.get_acc(sess_op=models.sess, preds_op=models.pred, batch_size=60, x_op=models.x_,
                               phase_train=models.phase_train)
            models.save_models('models/{}.ckpt'.format(step))
            print acc


