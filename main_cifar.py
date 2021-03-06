import tensorflow as tf
import numpy as np
import glob
import configure as cfg
from models import Models
from utils import show_progress
from eval import Eval
from aug import aug_lv3

from utils import plot_images , cls2onehot
from cifar_input import *
if __name__ == '__main__':
    # dp == DataProvider

    #download_data_url(url , './cifar_10')
    train_filenames=glob.glob('cifar_10/cifar-10-batches-py/data_batch*')
    test_filenames=glob.glob('cifar_10/cifar-10-batches-py/test_batch*')

    train_imgs , train_labs = get_images_labels(*train_filenames)
    test_imgs, test_labs = get_images_labels(*test_filenames)

    train_imgs = train_imgs/255.
    test_imgs = test_imgs / 255.
    train_labs = cls2onehot(train_labs, depth=10)
    test_labs = cls2onehot(test_labs, depth=10)
    train_imgs = train_imgs
    test_imgs = test_imgs
    # Setting models
    models = Models(n_classes = 10 , img_shape = (32,32,3))
    # Get batch xs , ys


    # Augmenatation
    # batch_xs = aug_lv3(batch_xs)
    # batch_xs = batch_xs / 255.
    # plot_images(batch_xs , batch_ys)

    # Training
    eval=Eval()
    for step in range(cfg.max_iter):
        show_progress(step , cfg.max_iter)
        batch_xs, batch_ys = next_batch(train_imgs, train_labs, 60)

        train_cost = models.training(batch_xs , batch_ys , cfg.lr)
        if step % cfg.ckpt  == 0 :
            print 'Validation ... '

            pred, pred_cls, eval_cost, accuracy = models.eval(test_imgs, test_labs)
            #pred_op, pred_cls, eval_cost, accuracy = models.eval(dp.val_imgs , dp.val_labs,)
            #acc = eval.get_acc(sess_op=models.sess, preds_op=models.pred[:,0], batch_size=60, x_op=models.x_,
            #                   phase_train=models.phase_train)
            print accuracy
            models.save_models('models/{}.ckpt'.format(step))
            print 'train cost : {}'.format(train_cost)
            print 'test cost : {}'.format(eval_cost)
            print 'test_acc : {}'.format(accuracy)



