import numpy as np
from PIL import Image
import click
import os
import os
import numpy as np
import math
import h5py
import scipy.io as sio
import scipy.misc
from keras.models import save_model, load_model, Model
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Activation, Subtract
import random
import matplotlib.pyplot as plt
from numpy import moveaxis

import cv2
from skimage.measure import compare_ssim,compare_psnr,compare_mse
import  matplotlib



def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def unet (input_size_1=256,input_size_2=256):
    inpt = Input(shape=(input_size_1, input_size_2, 1))

    conv1 = Conv2d_BN(inpt, 64, (3, 3))
    conv1 = Conv2d_BN(conv1,64, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 128, (3, 3))
    conv2 = Conv2d_BN(conv2, 128, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 256, (3, 3))
    conv3 = Conv2d_BN(conv3, 256, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 512, (3, 3))
    conv4 = Conv2d_BN(conv4, 512, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 1024, (3, 3))
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2d_BN(conv5, 512, (3, 3))
    conv5 = Dropout(0.5)(conv5)

    convt1 = Conv2dT_BN(conv5, 512, (3, 3))
    concat1 = concatenate([conv4, convt1], axis=3)
    concat1 = Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1, 512, (3, 3))
    conv6 = Conv2d_BN(conv6, 512, (3, 3))

    convt2 = Conv2dT_BN(conv6, 256, (3, 3))
    concat2 = concatenate([conv3, convt2], axis=3)
    concat2 = Dropout(0.5)(concat2)
    conv7 = Conv2d_BN(concat2, 256, (3, 3))
    conv7 = Conv2d_BN(conv7, 256, (3, 3))

    convt3 = Conv2dT_BN(conv7, 128, (3, 3))
    concat3 = concatenate([conv2, convt3], axis=3)
    concat3 = Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 128, (3, 3))
    conv8 = Conv2d_BN(conv8, 128, (3, 3))

    convt4 = Conv2dT_BN(conv8, 64, (3, 3))
    concat4 = concatenate([conv1, convt4], axis=3)
    concat4 = Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 64, (3, 3))
    conv9 = Conv2d_BN(conv9, 64, (3, 3))
    conv9 = Dropout(0.5)(conv9)
    outpt = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv9)

    model = Model(inpt, outpt)
    return model
def DnCNN(depth=25, filters=64, image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(256, 256, image_channels), name='input' + str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='conv' + str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth - 2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='conv' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn' + str(layer_count))(x)
        layer_count += 1
        x = Activation('relu', name='relu' + str(layer_count))(x)
        # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same', use_bias=False, name='conv' + str(layer_count))(x)
    layer_count += 1
    # x = Subtract(name='subtract' + str(layer_count))([inpt, x])  # input - noise
    model = Model(inputs=inpt, outputs=x, name='DnCNN')

    return model
# model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mse'])
# model.summary()

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
    output_dir_1 = 'E:/chenyalei/mocogan_v2/scripts/test_human_dncnn_1015/'
    # g = load_model(weight_path)
    g = DnCNN()
    g.load_weights(weight_path)
    # zero = np.zeros(shape=(1980,224,224,2))
    # for image_name in os.listdir(input_dir):
	# 	image = np.array([preprocess_image(load_image(os.path.join(input_dir, image_name)))])
	# 	x_test = image
	# 	generated_images = g.predict(x=x_test)
	# 	# generated_images = np.array(generated_images)
	# 	# print('shape',generated_images.shape)
	# 	generated = np.array([deprocess_image(img) for img in generated_images])
	# 	x_test = deprocess_image(x_test)
	# 	for i in range(generated_images.shape[0]):
	# 		x = x_test[i, :, :, :]
	# 		img = generated[i, :, :, :]
	# 		# print('img',img.shape)
	#
	# 		# output = np.concatenate((x, img), axis=1)
	# 		output = img
	# 		im = Image.fromarray(output.astype(np.uint8))
	# 		im.save(os.path.join(output_dir, image_name))
	# 		zero[i,:,:,:] = img
	#
	# 	np.save(os.path.join(output_dir_1, 'image_shape_1.npy'),zero)
    # x_test = np.zeros(shape=(86, 256, 256, 2))
    # # y_test = np.zeros(shape=(86, 256, 256, 2))
    # x_test = np.zeros(shape=(256, 256, 130, 2))
    # y_test = np.zeros(shape=(256, 256, 130, 2))
    # mat_A = h5py.File(input_dir)
    # mat_B = h5py.File('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_testGTData.mat')
    # # mat_A = sio.loadmat(input_dir)
    # # mat_B = sio.loadmat('E:/chenyalei/deblur-gan-master/real_imag_dataset/real_imag_testGTData.mat')
    # imgs_A1 = mat_A['real_data']
    # imgs_A2 = mat_A['imag_data']
    # imgs_B1 = mat_B['real_data']
    # imgs_B2 = mat_B['imag_data']
    # # imgs_A = mat_A['H']
    # # imgs_B = mat_B['H']
    # x_test[:, :, :, 0] = imgs_A1
    # x_test[:, :, :, 1] = imgs_A2
    # y_test[:, :, :, 0] = imgs_B1
    # y_test[:, :, :, 1] = imgs_B2
    # x_test = moveaxis(moveaxis(x_test, 2, 0),1,2)
    # y_test = moveaxis(moveaxis(y_test, 2, 0),1,2)
    # y_test = deprocess_image(y_test)
    # x_test = np.load('E:/chenyalei/u-net-master/u-net-master/unet_test_pred_531.npy')
    # y_test = np.load('E:/chenyalei/u-net-master/u-net-master/unet_testGT_531.npy')

    # x_test = np.load('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/test_motion_image.npy')
    # y_test = np.load('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/test_gt_image.npy')
    # rootdir1 = '/home/user/data1/mouse_data/FullBrain/test/gt/'
    # rootdir2 = '/home/user/data1/mouse_data/FullBrain/test/motion/'
    ####################################################################################################################
    # rootdir1 = 'E:/chenyalei/testhuman/hcp_motion_motionnum24_18_1.mat'
    # rootdir2 = 'E:/chenyalei/testhuman/hcp_gt_18.mat'
    #
    # test_data_1 = sio.loadmat(rootdir1)
    # test_data_2 = sio.loadmat(rootdir2)
    # x_test = abs(test_data_1['v_img3d'])
    # y_test = test_data_2['motion']

    x_test = np.load('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/hcp_motion_1-3.npy')
    y_test = np.load('E:/chenyalei/deblur-gan-master/dataset/GT_data_611/hcp_GT_1-3.npy')
    # list1 = os.listdir(rootdir1)
    # list2 = os.listdir(rootdir2)
    # list1.sort(key=lambda x: int(x[24:-4]))
    # list2.sort(key=lambda x: int(x[20:-4]))
    # x_test = np.zeros(shape=(1, 256, 256), dtype='float64')
    # y_test = np.zeros(shape=(1, 256, 256), dtype='float64')
    # for num in range(1, 2):
    #     path1 = os.path.join(rootdir1, list1[num])
    #     motion_img = sio.loadmat(path1)
    #     motion_img = abs(motion_img['data1'])
    #
    #     x_test = np.concatenate((x_test,motion_img),axis=0)
    #
    #     path2 = os.path.join(rootdir2, list2[num])
    #     GT1 = sio.loadmat(path2)
    #     GT1 = GT1['data2']
    #     y_test = np.concatenate((y_test, GT1), axis=0)
    # img_motion = Image.open('E:/chenyalei/humantestforpaper/human_simu_motion_1.png')
    # img_motion = img_motion.convert('L')
    # img_motion = np.array(img_motion)
    # img_motion = cv2.resize(img_motion,(256,256))
    # img_motion = (img_motion - np.min(img_motion)) / (np.max(img_motion) - np.min(img_motion))    # img1 = sio.loadmat(rootdir1)
    # # img1 = abs(img1['data'])
    # # img2 = sio.loadmat(rootdir2)
    # # img2 = img2['data']
    # x_test = img_motion
    # y_test = y_test[1:11, :, :]

    for i in range(x_test.shape[0]):

            x_test[i, :, :] = cv2.normalize(x_test[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
            # M = cv2.getRotationMatrix2D((x_test.shape[1]/2,x_test.shape[2]/2),90,1)
            # x_test[i, :, :] = cv2.warpAffine(x_test[i, :, :],M,(x_test.shape[1],x_test.shape[2]))
            y_test[i, :, :] = cv2.normalize(y_test[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)

    x_test = np.reshape(x_test, (x_test.shape[0], 256, 256, 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 256, 256, 1))
    #####################################################################################################################
    # realTestpath1 = '/home/user/data1/mouse_data/TestRatMatData/TestRAREImage3.mat'
    # realTestpath2 = '/home/user/data1/mouse_data/TestRatMatData/TestRAREImage2.mat'
    # realTestdata1 = sio.loadmat(realTestpath1)
    # x_test = realTestdata1['ImagData']
    # realTestdata2 = sio.loadmat(realTestpath2)
    # y_test = realTestdata2['ImagData']
    #
    #
    # # x_test_1 = realTestdata[:,:,0,:]
    # # x_test_2 = realTestdata[:,:,1,:]
    # # x_test = np.concatenate((x_test_1,x_test_2),axis=2)
    # x_test = moveaxis(moveaxis(x_test,0,2),0,1)
    # y_test = moveaxis(moveaxis(y_test,0,2),0,1)
    #
    #
    # for i in range(x_test.shape[0]):
    #         x_test[i, :, :] = cv2.normalize(x_test[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
    #         y_test[i, :, :] = cv2.normalize(y_test[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
    #
    # x_test = np.reshape(x_test, (x_test.shape[0], 256, 256, 1))
    # y_test = np.reshape(y_test, (y_test.shape[0], 256, 256, 1))
    ######################################################################################################################
    # img1 = io.imread('test_data3/JPEGImages/' + input_name[rand_num]).astype("float")
    # img2 = io.imread('test_data3/TargetImages/' + input_name[rand_num]).astype("float")
    # img1 = loadmat_motion('E:/chenyalei/deblur-gan-master/test_motion_424.mat')
    # img2 = loadmat_gt('E:/chenyalei/deblur-gan-master/test_gt_424.mat')
    # img1 = np.load('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/hcp_motion_1-3.npy')
    # img2 = np.load('E:/chenyalei/deblur-gan-master/dataset/GT_data_611/hcp_GT_1-3.npy')
    # img1 = np.load('E:/chenyalei/DATASET single/motionnum16/hcp_motion_13_17.npy')
    # img2 = np.load('E:/chenyalei/DATASET single/HCP_GT_13-17/hcp_GT_13_17.npy')


    # x_test = load_mat('E:/chenyalei/deblur-gan-master/dataset/testdata.mat')
    # y_test = load_mat('E:/chenyalei/deblur-gan-master/dataset/testgt.mat')

    generated_images = g.predict(x=x_test)

    # np.save('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/test_motion_image.npy',
    #         x_test)
    # np.save('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/test_gt_image.npy',
    #         y_test)
    # generated_images = np.array(generated_images)
    # print('shape',generated_images.shape)
    # generated = np.array([deprocess_image(img) for img in generated_images])
    generated = np.array(generated_images)
    np.save(os.path.join(output_dir_1, 'human_motion_test_dncnn.npy'), x_test)
    np.save(os.path.join(output_dir_1, 'human_gt_test_dncnn.npy'), y_test)
    np.save(os.path.join(output_dir_1, 'human_pred_test_dncnn.npy'), generated)
    # np.save('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/generated_image_epoch120.npy',
    #         generated)
    # pred_X = moveaxis(generated, 1, 2)
    # test_Y = moveaxis(y_test, 1, 2)
    # test_X = moveaxis(x_test, 1, 2)
    # dimg1 = generated[1, :, :, 0]
    # dimg2 = y_test[1, :, :, 0]
    # dimg3 = x_test[1, :, :, 0]
    # plt.figure(figsize=(6, 2))
    # plt.subplot(1, 3, 1)
    # plt.imshow(dimg1, cmap='gray')
    # plt.axis('off')
    # plt.title('pred')  # 不显示坐标
    # # plt.show()
    # # plt.figure('test_gt')
    # plt.subplot(1, 3, 2)
    # plt.imshow(dimg2, cmap='gray')
    # plt.axis('off')  # 不显示坐标轴
    # plt.title('test_gt')
    # # plt.show()
    # plt.subplot(1, 3, 3)
    # # plt.figure('test_motion')
    # plt.imshow(dimg3, cmap='gray')
    # plt.axis('off')  # 不显示坐标轴
    # plt.title('test_motion')
    # plt.show()
    ssim_ = []
    psnr_ = []
    mse_ = []
    #
    for i in range(generated.shape[0]):

        dimg1 = generated[i, :, :, 0]
        # dimg1 = dimg1.astype("float64")
        dimg2 = y_test[i, :, :, 0]
        dimg3 = x_test[i, :, :, 0]
        ssim = compare_ssim(dimg1, dimg2, multichannel=False)
        psnr = compare_psnr(dimg1, dimg2)
        mse = compare_mse(dimg1, dimg2)
        print(ssim, psnr, mse)
        ssim_.append(ssim)
        psnr_.append(psnr)
        mse_.append(mse)

        # d = cv2.absdiff(dimg1, dimg2)
        # # fig = plt.figure(dpi=200)
        # plt.figure(num=1)
        # cnorm = matplotlib.colors.Normalize(vmin=0, vmax=0.65)
        # m = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=matplotlib.cm.jet)
        # # d = np.abs(np.abs(img) - np.abs(img_))
        # m.set_array(d)
        # plt.imshow(d, norm=cnorm, cmap="jet")
        # plt.axis("off")
        # plt.colorbar(m)
        # # plt.show(m)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('/home/user/data1/yalei/mocoGAN/moco-gan-master/scripts/test_mouse_0804_sub/%s_%d_%s.png' % ('sub', i,"motion_fse"), bbox_inches='tight', dpi=400,
        #             pad_inches=0)

        dimg  = np.concatenate((dimg2,dimg1,dimg3),axis=1)
        plt.figure(num=2)
        plt.imshow(dimg,cmap='gray')
        plt.axis("off")

        # plt.show()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('E:/chenyalei/mocogan_v2/scripts/test_human_dncnn_1015/%s_%d_%s.png' % ('unet_epoch70', i, "human"), bbox_inches='tight', dpi=400,
                    pad_inches=0)
        # dimg  = Image.fromarray(dimg.astype(np.uint8),mode='GARY')
        # output_dir = '/home/user/data1/yalei/mocoGAN/moco-gan-master/scripts/test_mouse_728/'
        # plt.imsave(os.path.join(output_dir,'im{}_{}.png'.format(str(1),i)),dimg,cmap='gray')
    ssim_mean = np.mean(ssim_)
    psnr_mean = np.mean(psnr_)
    mse_mean = np.mean(mse_)
    print(ssim_mean, psnr_mean, mse_mean)
    # dimg1 = deprocess_image(dimg1)
    # dimg2 = deprocess_image(dimg2)
    # dimg3 = deprocess_image(dimg3)
    # ii = 0
    # dimg1 = generated[8, :, :, 0]
    # dimg2 = y_test[8, :, :, 0]
    # dimg3 = x_test[8, :, :, 0]

    # np.save(os.path.join(output_dir_1, 'DnCNN_testMotion_524_50epoch.npy'), x_test)
    # np.save(os.path.join(output_dir_1,'DnCNN_testpred_524_50epoch.npy'),generated)
    # np.save(os.path.join(output_dir_1,'DnCNN_testGT_524_50epoch.npy'),y_test)

    # x_test = deprocess_image(x_test)
    # y_test = deprocess_image(y_test)
    # for i in range(generated_images.shape[0]):
    # 	x1 = x_test[i, :, :, 0]
    # 	x2 = x_test[i, :, :, 1]
    # 	y1 = y_test[i, :, :, 0]
    # 	y2 = y_test[i, :, :, 1]
    #
    # 	img1 = generated[i, :, :, 0]
    # 	img2 = generated[i, :, :, 1]
    # 	# maxVal1=np.max(np.max(x1))
    # 	# maxVal2=np.max(np.max(x2))
    # 	# maxVal3=np.max(np.max(img1))
    # 	# maxVal4=np.max(np.max(img2))
    # 	# maxVal5=np.max(np.max(y1))
    # 	# maxVal6=np.max(np.max(y2))
    #
    # 	# print('img',img.shape)
    #
    # 	# output = np.concatenate((x, img), axis=1)
    # 	output = np.concatenate((x1, x2,img1,img2,y1,y2), axis=1)
    # 	output = Image.fromarray(output.astype(np.uint8))
    #
    # 	plt.imsave(os.path.join(output_dir,'im{}_{}.png'.format(str(1),i)),output,cmap='gray')

		# im = Image.fromarray(output.astype(np.uint8))
		# plt.imshow(im)
		# im.save(os.path.join(output_dir,str(i)+'.png'),im)
		# zero[i,:,:,:] = img

	# np.save(os.path.join(output_dir_1, 'image_shape_1.npy'),zero)




@click.command()
@click.option('--weight_path', default='E:/chenyalei/mocogan_v2/scripts/unet_weight/1013/unet_99_0.h5',help='Model weight')
@click.option('--input_dir', default='E:/chenyalei/deblur-gan-master/test_motion_424.mat',help='Image to deblur')
@click.option('--output_dir', default='E:/chenyalei/deblur-gan-master/images/output427',help='Deblurred image')
def deblur_command(weight_path, input_dir, output_dir):

    return deblur(weight_path, input_dir, output_dir)

# weight_path = 'E:/chenyalei/deblur-gan-master/weights/1116/generator_99_54.h5'
# input_dir = 'E:/chenyalei/deblur-gan-master/images/test/A'
# output_dir = 'E:/chenyalei/deblur-gan-master/images/output1116'


if __name__ == "__main__":
	deblur_command()
