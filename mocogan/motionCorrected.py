import numpy as np
from PIL import Image
import click
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import math
import h5py
import scipy.io as sio
import scipy.misc
from keras.models import save_model, load_model, Model
import random
import matplotlib.pyplot as plt
from numpy import moveaxis
import cv2
from skimage.measure import compare_ssim,compare_psnr,compare_mse
import  matplotlib

from skimage.transform import resize
from deblurgan.model_res import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image

# def loadmat_motion(input_dir):
#     mat= sio.loadmat(input_dir)
#     # imgs_A1 = mat['real_data']
#     # imgs_A2 = mat['imag_data']
#     imgs_train = mat['testdata']
#     # imgs_train = imgs_train['motion1']
#     # x_train = np.zeros(shape=(1410, 256, 256, 2))
#     # x_train[:, :, :, 0] = imgs_A1
#     # x_train[:, :, :, 1] = imgs_A2
#     imgs_train = moveaxis(imgs_train, 1, 2)
#     imgs_train = abs(imgs_train)
#     return imgs_train

def load_mat(input_dir):
    mat= sio.loadmat(input_dir)
    # imgs_A1 = mat['real_data']
    # imgs_A2 = mat['imag_data']
    imgs_train = mat['H']
    imgs_train = np.reshape(imgs_train,(imgs_train.shape[0],imgs_train.shape[1], imgs_train.shape[2], 1))

    # imgs_train = imgs_train['motion1']
    # x_train = np.zeros(shape=(1410, 256, 256, 2))
    # x_train[:, :, :, 0] = imgs_A1
    # x_train[:, :, :, 1] = imgs_A2
    # imgs_train = moveaxis(imgs_train, 1, 2)
    imgs_train = abs(imgs_train)
    return imgs_train

# def loadmat_gt(input_dir):
#     mat= sio.loadmat(input_dir)
#     # imgs_A1 = mat['real_data']
#     # imgs_A2 = mat['imag_data']
#     imgs_train = mat['testdata']
#     # imgs_train = imgs_train['motion1']
#     # x_train = np.zeros(shape=(1410, 256, 256, 2))
#     # x_train[:, :, :, 0] = imgs_A1
#     # x_train[:, :, :, 1] = imgs_A2
#     imgs_train = moveaxis(imgs_train, 1, 2)
#     # imgs_train = abs(imgs_train)
#     return imgs_train

# def batch_data_test(batch_size=8, input_size_1=256, input_size_2=256):
#     rand_num = random.randint(0, 99)
#     # img1 = io.imread('test_data3/JPEGImages/' + input_name[rand_num]).astype("float")
#     # img2 = io.imread('test_data3/TargetImages/' + input_name[rand_num]).astype("float")
#     # img1 = loadmat_gt('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_testMotionData.mat')
#     # img2 = loadmat_gt('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_testGTData.mat')
#     # img1 = np.load('E:/chenyalei/DATASET single/motionnum16/hcp_motion_13_17.npy')
#     # img2 = np.load('E:/chenyalei/DATASET single/HCP_GT_13-17/hcp_GT_13_17.npy')
#     rootdir1 = '/home/user/data1/mouse_data/DataSetRat/test/gt/MotionRatData_RARE201.mat'
#     rootdir2 = '/home/user/data1/mouse_data/DataSetRat/test/motion/gtRatData_RARE_201.mat'
#     img1 = sio.loadmat(rootdir1)
#     img1 = abs(img1['data'])
#     img2 = sio.loadmat(rootdir2)
#     img2 = img2['data']
#     # img1 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_10.npy')
#     # img2 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_10.npy')
#
#     img1 = cv2.normalize(img1, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
#     img2 = cv2.normalize(img2, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
#
#     # for i in range(img1.shape[0]):
#     #     img1[i,:,:] = (img1[i,:,:] - 127.5)/127.5
#     #     img2[i,:,:] = (img2[i,:,:] - 127.5)/127.5
#     img1 = img1[rand_num,:,:]
#     img2 = img2[rand_num,:,:]
#     # img1 = resize(img1, [input_size_1, input_size_2, 1])
#     # img2 = resize(img2, [input_size_1, input_size_2, 1])
#     img1 = np.reshape(img1, (1, input_size_1, input_size_2, 1))
#     img2 = np.reshape(img2, (1, input_size_1, input_size_2, 1))
#     # img1 /= 255
#     # img2 /= 255
#     batch_input = img1
#     batch_output = img2
#     for batch_iter in range(1, batch_size):
#         rand_num = random.randint(0, 99)
#         rootdir1 = '/home/user/data1/mouse_data/DataSetRat/test/gt/MotionRatData_RARE201.mat'
#         rootdir2 = '/home/user/data1/mouse_data/DataSetRat/test/motion/gtRatData_RARE_201.mat'
#         img1 = sio.loadmat(rootdir1)
#         img1 = abs(img1['data'])
#         img2 = sio.loadmat(rootdir2)
#         img2 = img2['data']
#
#         # img1 = io.imread('test_data3/JPEGImages/' + input_name[rand_num]).astype("float")
#         # img2 = io.imread('test_data3/TargetImages/' + input_name[rand_num]).astype("float")
#         # img1 = loadmat_motion('E:/chenyalei/deblur-gan-master/test_motion_424.mat')
#         # img2 = loadmat_gt('E:/chenyalei/deblur-gan-master/test_gt_424.mat')
#         # img1 = np.load('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/hcp_motion_1-3.npy')
#         # img2 = np.load('E:/chenyalei/deblur-gan-master/dataset/GT_data_611/hcp_GT_1-3.npy')
#         # img1 = np.load('E:/chenyalei/DATASET single/motionnum16/hcp_motion_13_17.npy')
#         # img2 = np.load('E:/chenyalei/DATASET single/HCP_GT_13-17/hcp_GT_13_17.npy')
#         img1 = cv2.normalize(img1, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
#         img2 = cv2.normalize(img2, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
#         # for i in range(img1.shape[0]):
#         #     img1[i, :, :] = (img1[i, :, :] - 127.5) / 127.5
#         #     img2[i, :, :] = (img2[i, :, :] - 127.5) / 127.5
#         img1 = img1[rand_num, :, :]
#         img2 = img2[rand_num, :, :]
#         # img1 = resize(img1, [input_size_1, input_size_2, 1])
#         # img2 = resize(img2, [input_size_1, input_size_2, 1])
#         img1 = np.reshape(img1, (1, input_size_1, input_size_2, 1))
#         img2 = np.reshape(img2, (1, input_size_1, input_size_2, 1))
#         # img1 /= 255
#         #         # img2 /= 255
#         # batch_input = np.concatenate((batch_input, img1), axis=0)
#         # batch_input_patch = np.zeros([batch_input.shape[0] * 4, 64, 64, 1], np.uint8)
#         batch_input = np.concatenate((batch_input, img1), axis=0)
#         batch_output = np.concatenate((batch_output, img2), axis=0)
#         # for i in range(batch_input.shape[0]):
#         #     batch_input_each = divide_method2(batch_input[i, :, :, :], 5, 5)
#         #     batch_input_patch[i:i + 4, :, :, :] = batch_input_each
#         # batch_output = np.concatenate((batch_output, img2), axis=0)
#         # batch_output_patch = np.zeros([batch_output.shape[0] * 4, 64, 64, 1], np.uint8)
#         # for i in range(batch_output.shape[0]):
#         #     batch_output_each = divide_method2(batch_output[i, :, :, :], 5, 5)
#         #     batch_output_patch[i:i + 4, :, :, :] = batch_output_each
#     return batch_input, batch_output

def deblur(weight_path, input_dir, output_dir):
    output_dir_1 = 'E:/chenyalei/mocogan_v2/scripts/test_1128/'
    # g = load_model(weight_path)
    g = generator_model()
    g.load_weights(weight_path)

    # x_train_1 = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/motionChannel1.npy')
    # x_train_1 = x_train_1[0:2000, :, :]
    # y_train_1 = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/motionfreeChannel1.npy')
    # y_train_1 = y_train_1[0:2000, :, :]
    #

    x_train_1 =  np.load('/home/user/data1/yalei/hcp_motion_1-3.npy')
    y_train_1 = np.load('/home/user/data1/yalei/hcp_GT_1-3.npy')

    # x_train_1 = sio.loadmat('/home/user/data2/bcx/real_rat_data/chaifen/mydata2-4-2.mat')
    # x_train_1 = sio.loadmat('/home/user/data2/bcx/real_rat_data/chaifen/mydata2-4-2.mat')
    # x_train_1 = x_train_1['data3_2']
    # x_train_1 = x_train_1.transpose(2,1,0)


    x_train = x_train_1
    y_train = y_train_1
    for i in range(x_train.shape[0]):
        x_train[i, :, :] = cv2.normalize(x_train[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
        y_train[i, :, :] = cv2.normalize(y_train[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)

    motion = np.float64(x_train)
    gt = np.float64(y_train)
    sio.savemat(os.path.join(output_dir, 'motion'), {'motion': motion})
    sio.savemat(os.path.join(output_dir, 'gt'), {'gt': gt})

    # sio.savemat(os.path.join(output_dir, 'mydata2-4-2'), {'motion': x_train})

    # plt.imshow(x_train[10,:,:], cmap='gray')
    # plt.show()

    # sio.savemat('./motion',{'motion':x_train})
    # sio.savemat('./gt',{'gt':y_train})

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], y_train.shape[2], 1))

    x_test = x_train
    y_test = y_train

    generated_images = g.predict(x=x_test)
    generated = np.array(generated_images)

    out = np.float64(generated)

    sio.savemat(os.path.join(output_dir, 'corrected'), {'corrected': out})





@click.command()
@click.option('--weight_path', default='/home/user/data2/bcx/mocogan_v2/scripts/weights_hcp/1217/generator_99_90.h5',help='Model weight')
@click.option('--input_dir', default='E:/chenyalei/deblur-gan-master/test_motion_424.mat',help='Image to deblur')
# @click.option('--output_dir', default='/home/user/data2/bcx/mocogan_v2/scripts/output',help='Deblurred image')
@click.option('--output_dir', default='/home/user/data2/bcx/mocogan_v2/scripts/output_hcp/traindata',help='Deblurred image')
def deblur_command(weight_path, input_dir, output_dir):

    return deblur(weight_path, input_dir, output_dir)

# weight_path = 'E:/chenyalei/deblur-gan-master/weights/1116/generator_99_54.h5'
# input_dir = 'E:/chenyalei/deblur-gan-master/images/test/A'
# output_dir = 'E:/chenyalei/deblur-gan-master/images/output1116'


if __name__ == "__main__":
	deblur_command()
