import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
import numpy as np
from keras.layers import MaxPooling2D
import tensorflow as tf
import cv2
import math
import torch

x_train_1 = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/motionChannel1.npy')
x_train_1 = x_train_1[0:20,:,:]
x_train = x_train_1
for i in range(x_train.shape[0]):
    x_train[i, :, :] = cv2.normalize(x_train[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)

y_pred = x_train[10, :, :]
y_pred = np.expand_dims(y_pred,axis=0)
y_pred = np.expand_dims(y_pred,axis=0)
y_pred = torch.from_numpy(y_pred)

tem = torch.nn.MaxPool2d((8,8))
y_pred = tem(y_pred)
entropy = []
grey_max = 255
grey_min = 0
y_pred = y_pred.numpy()
img = y_pred[0,0,:,:]
# img = torch.from_numpy(img)
img_max = np.max(img)
img_min = np.min(img)
img_grey = (grey_max-grey_min)*(img-img_min)/(img_max-img_min)+grey_min
BjMax = 0
E = 0
for m in range(32):
    for n in range(32):
        Bj = img_grey[m,n]
        Bj2 = math.pow(Bj,2)
        BjMax = BjMax+Bj2
BjMax = math.sqrt(BjMax)
for p in range(32):
    for q in range(32):
        Bj = img_grey[p,q]
        if Bj == 0:
            E_temp = 0
        else:
            # E_temp = 1
            E_temp = (Bj / BjMax) * math.log(Bj / BjMax)

        E = E+E_temp
E = -E
entropy.append(E)
entropy_mean = tf.reduce_mean(entropy)