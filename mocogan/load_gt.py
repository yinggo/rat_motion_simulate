import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import  matplotlib
from PIL import Image
import argparse
import scipy.io as sio
from deblurgan.utils import load_images, write_log,loadmat_motion,loadmat_gt,loadmat_c,loadmat_mouse
from numpy import moveaxis

rootdir1  = '/home/user/data1/mouse_data/DataSetRat/gt/'
rootdir2  = '/home/user/data1/mouse_data/DataSetRat/motion/'
list1 = os.listdir(rootdir1)
list2 = os.listdir(rootdir2)
list1.sort(key= lambda x:int(x[19:-4]))
list2.sort(key= lambda x:int(x[15:-4]))
y_train_1 = np.zeros(shape=(3000, 256, 256),dtype='float64')
x_train_1 = np.zeros(shape=(3000, 256, 256),dtype='float64')
y_train_2 = np.zeros(shape=(19000, 256, 256),dtype='float64')
x_train_2 = np.zeros(shape=(19000, 256, 256),dtype='float64')
for num in range(0,10):
    path1 = os.path.join(rootdir1,list1[num])
    motion_img = sio.loadmat(path1)
    motion_img = abs(motion_img['data'])

    x_train_1[num * 300:(num + 1) * 300, :, :] = motion_img
    path2 = os.path.join(rootdir2,list2[num])
    GT1 = sio.loadmat(path2)
    GT1 = GT1['data']
    y_train_1[num * 300:(num + 1) * 300, :, :] = GT1
for num in range(10, len(list1)):
    path1 = os.path.join(rootdir1, list1[num])
    motion_img = sio.loadmat(path1)
    motion_img = motion_img['data']
    x_train_2[num * 100:(num + 1) * 100, :, :] = motion_img
    path2 = os.path.join(rootdir2, list2[num])
    GT1 = sio.loadmat(path2)
    GT1 = GT1['data']
    y_train_2[num * 100:(num + 1) * 100, :, :] = GT1
x_train = np.concatenate((x_train_1, x_train_2),axis=0)
y_train = np.concatenate((y_train_1, y_train_2),axis=0)
    # plt.imshow(GT1[0,:,:],cmap='gray')
    # plt.show()




    # plt.imshow(GT1[50:-50, :, :], cmap='gray')
    # plt.show()

# GT_RAT = np.concatenate((GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1),axis=0)
# np.save('/home/user/data1/mouse_data/GT_RatData/rat_GT_11.npy',GT_RAT)


# plt.imshow(GT_RAT[0,:,:],cmap='gray')
# plt.show()
rootdir = '/home/user/data1/mouse_data/MotionRatData/rat_motion_10/'
list = os.listdir(rootdir)
# path = os.path.join(rootdir,list[0])
# noisy_motion_ = loadmat_c(path)
rat_motion_1 = np.zeros((1500,256,256),'float64')#列出文件夹下所有的目录与文件
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    motion_img = loadmat_mouse(path)
    motion_img = moveaxis(moveaxis(motion_img,0,2),0,1)
    rat_motion_1[i*50:(i+1)*50,:,:] = motion_img
def rotate(array):
    temp = np.zeros_like(array.transpose())
    for j in range(len(array)):
        for i in range(len(array[0])):
            temp[i][j] = array[j][len(array[0])-i-1]
    return temp
# motion_image = np.zeros(rat_motion_1.shape, np.float64)
# for i in range(rat_motion_1.shape[0]):
#     motion1=rotate(rat_motion_1[i,:,:])
#     motion_image[i,:,:] = motion1
# plt.imshow(rat_motion_1[0,:,:],cmap='gray')
# plt.show()
np.save('/home/user/data1/mouse_data/MotionRatData/rat_motion_10.npy',rat_motion_1)
# a = hcp_motion_13_17[0,:,:]
# GT = sio.loadmat('E:/chenyalei/DATASET single/HCP_GT_13-17/HCP_brain13.mat')
# GT = GT['hcp']
# GT = GT[70:100,:,:]
# hcp_GT_1 = np.concatenate((GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT,GT),axis=0)
# # #
# GT1 = sio.loadmat('E:/chenyalei/DATASET single/HCP_GT_13-17/HCP_brain14.mat')
# GT1 = GT1['hcp']
# GT1 = GT1[70:100,:,:]
# hcp_GT_2 = np.concatenate((GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1,GT1),axis=0)
# # #
# GT2 = sio.loadmat('E:/chenyalei/DATASET single/HCP_GT_13-17/HCP_brain15.mat')
# GT2 = GT2['hcp']
# GT2 = GT2[70:100,:,:]
# hcp_GT_3 = np.concatenate((GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2,GT2),axis=0)
#
# GT3 = sio.loadmat('E:/chenyalei/DATASET single/HCP_GT_13-17/HCP_brain16.mat')
# GT3 = GT3['hcp']
# GT3 = GT3[70:100,:,:]
# hcp_GT_4 = np.concatenate((GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3,GT3),axis=0)
#
# GT4 = sio.loadmat('E:/chenyalei/DATASET single/HCP_GT_13-17/HCP_brain17.mat')
# GT4 = GT4['hcp']
# GT4 = GT4[70:100,:,:]
# hcp_GT_5 = np.concatenate((GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4,GT4),axis=0)
#
# hcp_GT_13_17 = np.concatenate((hcp_GT_1,hcp_GT_2,hcp_GT_3,hcp_GT_4,hcp_GT_5),axis=0)
# np.save('E:/chenyalei/DATASET single/HCP_GT_13-17/hcp_GT_13_17.npy',hcp_GT_13_17)
path1 = 'E:/chenyalei/DATASET single/knee_gt_data/knee_data_256_51.mat'
knee_motion_1 = loadmat_c(path1)
knee_motion_1 = np.concatenate((knee_motion_1,knee_motion_1,knee_motion_1,knee_motion_1,knee_motion_1,knee_motion_1,knee_motion_1,
                                knee_motion_1,knee_motion_1,knee_motion_1,knee_motion_1,knee_motion_1,knee_motion_1,knee_motion_1,knee_motion_1),axis=0)
path2 = 'E:/chenyalei/DATASET single/knee_gt_data/knee_data_256_52.mat'
knee_motion_2 = loadmat_c(path2)
knee_motion_2 = np.concatenate((knee_motion_2,knee_motion_2,knee_motion_2,knee_motion_2,knee_motion_2,knee_motion_2,knee_motion_2,
                                knee_motion_2,knee_motion_2,knee_motion_2,knee_motion_2,knee_motion_2,knee_motion_2,knee_motion_2,knee_motion_2),axis=0)
path3 = 'E:/chenyalei/DATASET single/knee_gt_data/knee_data_256_53.mat'
knee_motion_3 = loadmat_c(path3)
knee_motion_3 = np.concatenate((knee_motion_3,knee_motion_3,knee_motion_3,knee_motion_3,knee_motion_3,knee_motion_3,knee_motion_3,
                                knee_motion_3,knee_motion_3,knee_motion_3,knee_motion_3,knee_motion_3,knee_motion_3,knee_motion_3,knee_motion_3),axis=0)
path4 = 'E:/chenyalei/DATASET single/knee_gt_data/knee_data_256_54.mat'
knee_motion_4 = loadmat_c(path4)
knee_motion_4 = np.concatenate((knee_motion_4,knee_motion_4,knee_motion_4,knee_motion_4,knee_motion_4,knee_motion_4,knee_motion_4,
                                knee_motion_4,knee_motion_4,knee_motion_4,knee_motion_4,knee_motion_4,knee_motion_4,knee_motion_4,knee_motion_4),axis=0)
path5 = 'E:/chenyalei/DATASET single/knee_gt_data/knee_data_256_55.mat'
knee_motion_5 = loadmat_c(path5)
knee_motion_5 = np.concatenate((knee_motion_5,knee_motion_5,knee_motion_5,knee_motion_5,knee_motion_5,knee_motion_5,knee_motion_5,
                                knee_motion_5,knee_motion_5,knee_motion_5,knee_motion_5,knee_motion_5,knee_motion_5,knee_motion_5,knee_motion_5),axis=0)
path6 = 'E:/chenyalei/DATASET single/knee_gt_data/knee_data_256_56.mat'
knee_motion_6 = loadmat_c(path6)
knee_motion_6 = np.concatenate((knee_motion_6,knee_motion_6,knee_motion_6,knee_motion_6,knee_motion_6,knee_motion_6,knee_motion_6,
                                knee_motion_6,knee_motion_6,knee_motion_6,knee_motion_6,knee_motion_6,knee_motion_6,knee_motion_6,knee_motion_6),axis=0)
path7 = 'E:/chenyalei/DATASET single/knee_gt_data/knee_data_256_57.mat'
knee_motion_7 = loadmat_c(path7)
knee_motion_7 = np.concatenate((knee_motion_7,knee_motion_7,knee_motion_7,knee_motion_7,knee_motion_7,knee_motion_7,knee_motion_7,
                                knee_motion_7,knee_motion_7,knee_motion_7,knee_motion_7,knee_motion_7,knee_motion_7,knee_motion_7,knee_motion_7),axis=0)
path8 = 'E:/chenyalei/DATASET single/knee_gt_data/knee_data_256_59.mat'
knee_motion_8 = loadmat_c(path8)
knee_motion_8 = np.concatenate((knee_motion_8,knee_motion_8,knee_motion_8,knee_motion_8,knee_motion_8,knee_motion_8,knee_motion_8,
                                knee_motion_8,knee_motion_8,knee_motion_8,knee_motion_8,knee_motion_8,knee_motion_8,knee_motion_8,knee_motion_8),axis=0)
knee_motion = np.concatenate((knee_motion_1,knee_motion_2,knee_motion_3,knee_motion_4,knee_motion_5,knee_motion_6,knee_motion_7,knee_motion_8),axis=0)
# # hcp_GT_1_3 = np.concatenate((hcp_GT_1,hcp_GT_2),axis=0)
# a = knee_motion_1[0,:,:]
# b = knee_motion_2[0,:,:]
# c = a-b
# plt.imshow(knee_motion_1[0,:,:],cmap='gray')
# plt.show()
# plt.imshow(knee_motion_2[0,:,:],cmap='gray')
# plt.show()
np.save('E:/chenyalei/deblur-gan-master/dataset/knee_data/knee_gt/knee_gt_629.npy',knee_motion)
# # a = GT_1[0,:,:]-GT_1[30,:,:]
rootdir1 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_13.mat'
hcp_motion1 = loadmat_c(rootdir1)
rootdir2 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_14.mat'
hcp_motion2 = loadmat_c(rootdir1)
rootdir3 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_15.mat'
hcp_motion3 = loadmat_c(rootdir1)
rootdir4 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_16.mat'
hcp_motion4 = loadmat_c(rootdir1)
rootdir5 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_17.mat'
hcp_motion5 = loadmat_c(rootdir1)
rootdir6 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_18.mat'
hcp_motion6 = loadmat_c(rootdir1)
rootdir7 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_19.mat'
hcp_motion7 = loadmat_c(rootdir1)
rootdir8 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_20.mat'
hcp_motion8 = loadmat_c(rootdir1)
rootdir9 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_21.mat'
hcp_motion9 = loadmat_c(rootdir1)
rootdir10 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_22.mat'
hcp_motion10 = loadmat_c(rootdir1)
rootdir11 = 'E:/chenyalei/DATASET single/hcp_motion_13-23/hcp_motion_23.mat'
hcp_motion11 = loadmat_c(rootdir1)
hcp_motion_13_23 = np.concatenate((hcp_motion1,hcp_motion2,hcp_motion3,hcp_motion4,hcp_motion5,hcp_motion6,hcp_motion7,hcp_motion8,hcp_motion9,hcp_motion10,hcp_motion11),axis=0)
# list = os.listdir(rootdir)
# path = os.path.join(rootdir,list[0])
# noisy_motion_ = loadmat_c(path)
# hcp_motion_13_23 = np.zeros((6000,256,256),'float64')#列出文件夹下所有的目录与文件
# for i in range(0,len(list)):
#     path = os.path.join(rootdir,list[i])
#     hcp_motion = loadmat_c(path)
#     print(hcp_motion_13_23[i*600:(i+1)*600,:,:].shape)
       # if os.path.isfile(path):

# gt = loadmat_motion('E:/chenyalei/DATASET single/noisy_data/noisy_motion/noisy_motion_')
# gt = gt.transpose(0,2,1,3)
def rotate(array):
    temp = np.zeros_like(array.transpose())
    for j in range(len(array)):
        for i in range(len(array[0])):
            temp[i][j] = array[j][len(array[0])-i-1]
    return temp
motion_image = np.zeros(hcp_motion_13_23.shape, np.float64)
for i in range(hcp_motion_13_23.shape[0]):
    motion1=rotate(hcp_motion_13_23[i,:,:])
    motion_image[i,:,:] = motion1
# motion_image_ = motion_image[0,:,:]
plt.imshow(motion_image[1,:,:],cmap='gray')
plt.show()
np.save('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/noisy_motion.npy',noisy_motion_)

plt.imshow(motion[31,:,:],cmap='gray')
plt.show()