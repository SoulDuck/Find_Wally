from image_processing import ImageProcessing
import numpy as np
class Eval():
    def __init__(self):
        self.img_processing= ImageProcessing()
        self.src_dir = 'test_imgs'
        self.imgs_coords = self.img_processing.generate_copped_imgs(self.src_dir , 32, 32, 64, 64)
        self.answer_sheet = {'1.png': [73, 74] , '2.png':[875 , 874] , '3.png':[146 ,147]}

        """
        best           second 
        # [1,33](73)   [1,32](72)    1.png
        # [13,43](875) [13,42](874) 3.png
        # [3,20](146)  [3,21](147)   2.png

        """

    def check_preds(self  , preds , answer_indices ):
        """
        :param imgs:
        :param coords:
        :param preds:
        :param answer_indices:
        :return:
        """
        max_idx = np.argmax(preds , axis =0 )
        max_value= np.max(preds , axis = 0 )

        # debug
        print 'max idx : {} , max value : {} , answer_indicies : {}'.format(max_idx , max_value , answer_indices)
        answer=False
        if max_value > 0.5:
            if max_idx in answer_indices:
                answer = True
            else:
                answer = False
        return answer

    def validate(self, imgs, sess_op, preds_op, batch_size, x_op, phase_train , normalize):
        batch_list = self.divide_images(imgs , batch_size)
        preds_list = []
        for batch_xs in batch_list:
            if normalize and np.max(batch_xs) > 1:
                batch_xs = batch_xs/255.

            preds = sess_op.run(preds_op ,  {x_op : batch_xs , phase_train : False })
            preds_list.extend(preds)
        print preds_list
        assert len(preds_list) == len(imgs)
        return preds_list

    def get_acc(self , sess_op , preds_op  , batch_size , x_op  , phase_train):
        answers = []
        for key in self.imgs_coords.keys():
            imgs , coords  = self.imgs_coords[key]
            answer_indices = self.answer_sheet[key]
            preds = self.validate(imgs, sess_op, preds_op, batch_size, x_op, phase_train , normalize=True)
            answer = self.check_preds(preds , answer_indices)
            answers.append(answer)
        return np.sum(answers) / float(len(answers))

    def divide_images(self, images, batch_size):
        debug_flag_lv0 = False
        debug_flag_lv1 = False
        if __debug__ == debug_flag_lv0:
            print 'debug start | utils.py | divide_images'
        batch_img_list = []
        share = len(images) / batch_size
        # print len(images)
        # print len(labels)
        # print 'share :',share

        for i in range(share + 1):
            if i == share:
                imgs = images[i * batch_size:]
                # print i+1, len(imgs), len(labs)
                batch_img_list.append(imgs)
                if __debug__ == debug_flag_lv1:
                    print "######utils.py: divide_images_from_batch debug mode#####"
                    print 'total :', len(images), 'batch', i * batch_size, '-', len(images)
            else:
                imgs = images[i * batch_size:(i + 1) * batch_size]
                # print i , len(imgs) , len(labs)
                batch_img_list.append(imgs)
                if __debug__ == debug_flag_lv1:
                    print "######utils.py: divide_images_from_batch debug mode######"
                    print 'total :', len(images), 'batch', i * batch_size, ":", (i + 1) * batch_size
        return batch_img_list
if __name__ == '__main__':
    eval=Eval()
    print eval.get_acc()



