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

# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)
tmp = []
for i in range(256):
    tmp.append(0)
val = 0
k = 0
res = 0
tmp = []

def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

# def entropy_loss(y_true,y_pred):
#     a = y_pred
#     with tf.Session():
#         y_pred = y_pred.eval()
#     entropy = []
#     for i in range(len(y_pred[0])):
#         img = y_pred[i,:,:,0]
#         for m in range(len(img)):
#             for n in range(len(img[i])):
#                 f_j = img[m][n]
#                 tmp[f_j] = float(tmp[val]+1)
#                 k = float(k+1)
#         for i in range(len(tmp)):
#             if(tmp[i] == 0):
#                 res = res
#             else:
#                 res = float(res - tmp[i] * (math.log(tmp[i])/math.log(2.0)))
#
#         entropy.append(res)
#     entropy_mean = np.mean(entropy)
#
#     return entropy_mean

def entropy_loss(y_true,y_pred):
    # y_pred = MaxPooling2D(pool_size=(8, 8))(y_pred)
    entropy = []
    grey_max = 255
    grey_min = 0
    for i in range(2):
        img = y_pred[i,0:4,0:4,0]
        img_max = tf.reduce_max(img)
        img_min = tf.reduce_min(img)
        img_grey = (grey_max-grey_min)*(img-img_min)/(img_max-img_min)+grey_min
        BjMax = 0
        E = 0
        for m in range(4):
            for n in range(4):
                Bj = img_grey[m,n]
                Bj2 = tf.pow(Bj,2)
                BjMax = BjMax+Bj2
        BjMax = tf.sqrt(BjMax)
        for p in range(4):
            for q in range(4):
                Bj = img_grey[p,q]
                if Bj == 0:
                    E_temp = 0
                else:
                    E_temp = (Bj / BjMax) * tf.math.log(Bj / BjMax)

                E = E+E_temp
        E = -E
        entropy.append(E)
    entropy_mean = tf.reduce_mean(entropy)

    return entropy_mean

def entropy_loss2(y_true,y_pred):
    entropy = []
    for i in range(2):
        img_grey = y_pred[i,:,:,0]
        E = 0
        ind = tf.where(img_grey > 0)
        img_new = tf.gather_nd(img_grey,ind)

        img_log = tf.math.log(img_new)
        out = img_new * img_log
        E = tf.reduce_sum(out)/(256*256)

        entropy.append(-E)
    entropy_mean = tf.reduce_mean(entropy)

    return entropy_mean

def my_entropy_loss(y_true,y_pred):
    entropy = []
    for i in range(2):
        with tf.Session():
            y_pred = y_pred.eval()
            img = y_pred[i,:,:,0]
            img_max = tf.reduce_max(img)
            img_min = tf.reduce_min(img)
            img = 255 * (img - img_min) / (img_max - img_min)
            tmp = []
            for i in range(256):
                tmp.append(0)
            val = 0
            k = 0
            res = 0
            for i in range(256):
                for j in range(256):
                    val = img[i][j]
                    val = tf.cast(val,tf.uint8)
                    tmp[val] = float(tmp[val] + 1)
                    k = float(k + 1)
            for i in range(len(tmp)):
                tmp[i] = float(tmp[i] / k)
            for i in range(len(tmp)):
                if (tmp[i] == 0):
                    res = res
                else:
                    res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
            entropy.append(res)
    entropy_mean = tf.reduce_mean(entropy)
    return entropy_mean

def entropy_l1_loss(y_true, y_pred):
    loss1 = entropy_loss2(y_true,y_pred)
    loss2 = K.mean(K.abs(y_true - y_pred))
    return 0.0*loss1+1.0*loss2

def edge_l1_loss(y_true, y_pred):
    loss1 = K.mean(K.abs(y_true - y_pred))
    y_true_edge = tf.image.sobel_edges(y_true)
    y_pred_edge = tf.image.sobel_edges(y_pred)
    loss2 = K.mean(K.abs(y_pred_edge - y_true_edge))
    return 0.8*loss1+0.2*loss2



def perceptual_loss_100(y_true, y_pred):
    return 100 * perceptual_loss(y_true, y_pred)

def perceptual_l1_loss(y_true,y_pred):

    return 0.8 * perceptual_loss(y_true, y_pred) + 0.2 * l1_loss(y_true, y_pred)


def perceptual_loss(y_true, y_pred):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False




    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


# def wasserstein_loss(y_true, y_pred):
#     return K.mean(y_true*y_pred)

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true*y_pred)


def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)
