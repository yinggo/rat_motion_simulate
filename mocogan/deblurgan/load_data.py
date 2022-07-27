import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import datetime
import click
import numpy as np
import tqdm



import matplotlib.pyplot as plt
import cv2
import scipy.io as sio


output_dir_1 = 'E:/chenyalei/mocogan_v2/data/'
rootdir1 = 'E:/chenyalei/mocogan_v2/data/markov_no_blur_data/motion/'
rootdir2 = 'E:/chenyalei/mocogan_v2/data/markov_no_blur_data/gt/'
list1 = os.listdir(rootdir1)
list2 = os.listdir(rootdir2)
list1.sort(key=lambda x: int(x[12:-4]))
list2.sort(key=lambda x: int(x[8:-4]))

y_train_1 = np.zeros(shape=(1, 256, 256), dtype='float64')
x_train_1 = np.zeros(shape=(1, 256, 256), dtype='float64')
# x_train_1 = np.array(x_train_1)
# y_train_1 = np.array(y_train_1)
# y_train_2 = np.zeros(shape=(19000, 256, 256), dtype='float64')
# x_train_2 = np.zeros(shape=(19000, 256, 256), dtype='float64')
# for num in range(0, len(list1)):
#     path1 = os.path.join(rootdir1, list1[num])
#     motion_img = sio.loadmat(path1)
#     motion_img = abs(motion_img['data'])
#
#     x_train_1[num * 300:(num + 1) * 300, :, :] = motion_img
#     path2 = os.path.join(rootdir2, list2[num])
#     GT1 = sio.loadmat(path2)
#     GT1 = GT1['data']
#     y_train_1[num * 300:(num + 1) * 300, :, :] = GT1
for num in range(0, len(list1)):
# for num in range(0, 1):
    path1 = os.path.join(rootdir1, list1[num])
    motion_img = sio.loadmat(path1)
    motion_img = abs(motion_img['data1'])

    x_train_1 = np.concatenate((x_train_1, motion_img), axis=0)

    path2 = os.path.join(rootdir2, list2[num])
    GT1 = sio.loadmat(path2)
    GT1 = GT1['data2']
    y_train_1 = np.concatenate((y_train_1, GT1), axis=0)
    # y_train_1.append(GT1)
x_train = x_train_1[1:, :, :]
y_train = y_train_1[1:, :, :]

np.save(os.path.join(output_dir_1, 'train_motion_markov_no_blur.npy'), x_train)
np.save(os.path.join(output_dir_1, 'train_gt_markov_no_blur.npy'), y_train)

# y_train_3c = np.zeros(shape=(x_train.shape[0], 256, 256,3), dtype='float64')
# x_train_3c = np.zeros(shape=(x_train.shape[0], 256, 256,3), dtype='float64')
# for i in range(x_train.shape[0]):
#     x_train_3c[i, :, :, 0] = x_train[i, :, :]
#     x_train_3c[i, :, :, 1] = x_train[i, :, :]
#     x_train_3c[i, :, :, 2] = x_train[i, :, :]
#     y_train_3c[i, :, :, 0] = y_train[i, :, :]
#     y_train_3c[i, :, :, 1] = y_train[i, :, :]
#     y_train_3c[i, :, :, 2] = y_train[i, :, :]
# print(x_train_3c.shape)
# np.save(os.path.join(output_dir_1, 'motionChannel3.npy'), x_train_3c)
# np.save(os.path.join(output_dir_1, 'motionfreeChannel3.npy'), y_train_3c)