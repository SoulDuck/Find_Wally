import Tester
import numpy as np
def cls2onehot(cls , depth):

    labs=np.zeros([len(cls) , depth])
    for i,c in enumerate(cls):
        labs[i,c]=1
    return labs
restore_model  = './models/model-3267'
tester=Tester.Tester(None)
tester._reconstruct_model(restore_model)
tester.n_classes =2

"""
best           second 
# [1,33](73)   [1,32](72)    1.png
# [13,43](875) [13,42](874) 3.png
# [3,20](146)  [3,21](147)   2.png

"""

test_imgs = np.load('./test_imgs/3.npy')
test_labs=[0]*len(test_imgs)

test_labs=cls2onehot(test_labs ,2 )


test_imgs = test_imgs/255.
tester.validate(test_imgs , test_labs, 60 ,0 ,False)

print np.where([np.asarray(tester.pred_all)[:,0] > 0.5])[1]

