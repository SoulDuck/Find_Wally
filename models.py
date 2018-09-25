from cnn import convolution2d , algorithm , gap
import tensorflow as tf
from aug import apply_aug_lv0 , aug_lv0
class Models(object):

    def __init__(self, n_classes , img_shape):

        self.n_classes = n_classes
        self.img_shape = img_shape
        self._define_placeholder()
        # Model
        self.simple_convnet([16,16,32,32,64] , [3,3,3,3,3] , [2,1,1,2,2] , self.n_classes )
        # trainer
        self.trainer('adam', False)
        # start Session
        self._start_sess()



    def _define_placeholder(self ):
        self.img_h, self.img_w, self.img_ch = self.img_shape
        self.x_ = tf.placeholder(dtype=tf.float32, shape=[None ,self.img_h,self.img_w,self.img_ch ] , name='x_')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_classes], name='y_')
        self.phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
        self.lr_ = tf.placeholder(dtype=tf.float32, name='learning_rate')

    def simple_convnet(self , out_chs, kernels , strides  , n_classes ):
        layer = apply_aug_lv0(self.x_ , aug_lv0, self.phase_train , crop_h = self.img_h , crop_w = self.img_w )
        layer = layer

        assert len(out_chs) == len(kernels) == len(strides)
        for i in range(len(out_chs)):
            layer = convolution2d('conv{}'.format(i) , layer , out_chs[i] , kernels[i] , strides[i] )
        top_conv = tf.identity(layer , 'top_conv')
        self.logits = gap('gap' , top_conv , n_classes)

    def trainer(self , optimizer , use_l2_loss):
        self.pred, self.pred_cls, self.cost, self.train_op, self.correct_pred, self.accuracy = algorithm(self.logits,
                                                                                                         self.y_,
                                                                                                         self.lr_,
                                                                                                         optimizer,
                                                                                                         use_l2_loss)
    def _start_sess(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=30)
        init=tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
        self.sess.run(init)

    def training(self , batch_xs , batch_ys ,lr):
        feed_dict = {self.x_:batch_xs , self.y_: batch_ys , self.phase_train: True , self.lr_: lr}
        fetches = [self.cost, self.train_op]
        cost , _ = self.sess.run(fetches , feed_dict)
        return cost

    def eval(self , val_xs , val_ys):
        feed_dict = {self.x_: val_xs, self.y_: val_ys, self.phase_train: False}
        fetches = [self.pred, self.pred_cls, self.cost , self.accuracy]
        pred, pred_cls, cost, accuracy = self.sess.run(fetches, feed_dict)
        return pred, pred_cls, cost, accuracy

    def save_models(self , paths):
        self.saver.save(self.sess , save_path = paths)


