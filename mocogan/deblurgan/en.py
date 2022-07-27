import cv2,math
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import keras.backend as K

sess = tf.Session()
def get_tensor():

    motion = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/motionChannel1.npy')
    motion = motion[0:2000, :, :]
    gt = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/motionfreeChannel1.npy')
    gt = gt[0:2000, :, :]
    for i in range(motion.shape[0]):
        motion[i, :, :] = cv2.normalize(motion[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
        gt[i, :, :] = cv2.normalize(gt[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)

    corrected = sio.loadmat('/home/user/data2/bcx/mocogan_v2/scripts/output_entropy_div_10000/corrected.mat')['corrected']

    x = motion[500, :, :]
    y = gt[500, :, :]
    z = np.squeeze(corrected[500,:,:])

    print(np.mean(np.abs(y-x)))
    print(np.mean(np.abs(y-z)))

    # plt.imshow(x,cmap='gray')
    # plt.show()
    #
    # plt.imshow(y, cmap='gray')
    # plt.show()
    #
    # plt.imshow(z, cmap='gray')
    # plt.show()

    x_tensor = tf.constant(x)
    # x_max = tf.reduce_max(x_tensor)
    # x_min = tf.reduce_min(x_tensor)
    # x_tensor = (x_tensor - x_min) / (x_max - x_min) + 0

    y_tensor = tf.constant(y)
    # y_max = tf.reduce_max(y_tensor)
    # y_min = tf.reduce_min(y_tensor)
    # y_tensor = (y_tensor - y_min) / (y_max - y_min) + 0

    z_tensor = tf.constant(z)
    # z_max = tf.reduce_max(z_tensor)
    # z_min = tf.reduce_min(z_tensor)
    # z_tensor = (z_tensor - z_min) / (z_max - z_min) + 0

    ind_x = tf.where(x_tensor > 0 )
    new_x = tf.gather_nd(x_tensor, ind_x)

    ind_y = tf.where(y_tensor > 0 )
    new_y = tf.gather_nd(y_tensor, ind_y)

    ind_z = tf.where(z_tensor > 0)
    new_z = tf.gather_nd(z_tensor, ind_z)

    log_x = tf.math.log(new_x)
    log_y = tf.math.log(new_y)
    log_z = tf.math.log(new_z)

    E_x = -tf.reduce_sum(new_x * log_x)
    E_y = -tf.reduce_sum(new_y * log_y)
    E_z = -tf.reduce_sum(new_z * log_z)

    # x = tf.random_uniform((5,4))
    # ind = tf.where(x > 0.5)
    # y = tf.gather_nd(x,ind)
    # z = tf.math.log(y)
    # out = y * z
    # E = tf.reduce_sum(out)
    # w = x ** 2

    # return new_x,new_y,new_z
    # return log_x,log_y,log_z
    return E_x,E_y,E_z

E_x,E_y,E_z= get_tensor()
E_x_,E_y_,E_z_= sess.run([E_x,E_y,E_z])
print(E_x_)
print(E_y_)
print(E_z_)
print(E_z_/(10000))


c = 1
# a = np.array([[1,2],[3,4],[0,5]])
# a = tf.convert_to_tensor(a)
# with tf.Session() as sess:
#     ind = tf.where(a != 0)
#     print(sess.run(ind))
#     output = tf.gather_nd(a,ind)
#     print(sess.run(output))

# tmp = []
# for i in range(256):
#     tmp.append(0)
# val = 0
# k = 0
# res = 0
# image = cv2.imread('circuit.tif',0)
#
# img = np.zeros(image.shape, dtype=np.float32)
# cv2.normalize(image, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# img_max = np.max(img)
# img_min = np.min(img)
# img = 255 * (img - img_min) / (img_max - img_min)
# img = np.uint8(img)
# for i in range(len(img)):
#     for j in range(len(img[i])):
#         val = img[i][j]
#         tmp[val] = float(tmp[val] + 1)
#         k =  float(k + 1)
# for i in range(len(tmp)):
#     tmp[i] = float(tmp[i] / k)
# for i in range(len(tmp)):
#     if(tmp[i] == 0):
#         res = res
#     else:
#         res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
# print(res)
