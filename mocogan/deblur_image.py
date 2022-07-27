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
import random
import matplotlib.pyplot as plt
from numpy import moveaxis
import cv2
from skimage.measure import compare_ssim,compare_psnr,compare_mse
import  matplotlib

from skimage.transform import resize
from deblurgan.model_res3c import generator_model
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
    output_dir_1 = 'E:/chenyalei/mocogan_v2/scripts/test_pre_l1_simu_0927/'
    # g = load_model(weight_path)
    g = generator_model()
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
    # rootdir1 = '/home/user/data1/mouse_data/TestRatMatData/TestRAREImage4.mat'
    # rootdir2 = '/home/user/data1/mouse_data/TestRatMatData/TestRAREImage2.mat'
    rootdir1 = 'E:/chenyalei/mousedata_test_simu/gt/'
    rootdir2 = 'E:/chenyalei/mousedata_test_simu/motion/'
    list1 = os.listdir(rootdir1)
    list2 = os.listdir(rootdir2)
    list1.sort(key=lambda x: int(x[24:-4]))
    list2.sort(key=lambda x: int(x[20:-4]))
    x_test = np.zeros(shape=(1, 256, 256), dtype='float64')
    y_test = np.zeros(shape=(1, 256, 256), dtype='float64')
    for num in range(0, 1):
        path1 = os.path.join(rootdir1, list1[num])
        motion_img = sio.loadmat(path1)
        motion_img = abs(motion_img['data1'])

        x_test = np.concatenate((x_test,motion_img),axis=0)

        path2 = os.path.join(rootdir2, list2[num])
        GT1 = sio.loadmat(path2)
        GT1 = GT1['data2']
        y_test = np.concatenate((y_test, GT1), axis=0)
    # # # img1 = sio.loadmat(rootdir1)
    # # # img1 = abs(img1['data'])
    # # # img2 = sio.loadmat(rootdir2)
    # # # img2 = img2['data']
    x_test = x_test[1:11, :, :]
    y_test = y_test[1:11, :, :]
    y_test_3c = np.zeros(shape=(y_test.shape[0], 256, 256, 3), dtype='float64')
    x_test_3c = np.zeros(shape=(x_test.shape[0], 256, 256, 3), dtype='float64')
    for i in range(x_test.shape[0]):
        x_test_3c[i, :, :, 0] = x_test[i, :, :]
        x_test_3c[i, :, :, 1] = x_test[i, :, :]
        x_test_3c[i, :, :, 2] = x_test[i, :, :]
        y_test_3c[i, :, :, 0] = y_test[i, :, :]
        y_test_3c[i, :, :, 1] = y_test[i, :, :]
        y_test_3c[i, :, :, 2] = y_test[i, :, :]
    ####################################################################################################################
    # realTestpath = '/home/user/data1/mouse_data/TestRatMatData/RAREImage8.mat'
    # realTestpath1 = '/home/user/data1/mouse_data/TestRatMatData/TestRAREImage4.mat'
    # realTestpath2 = '/home/user/data1/mouse_data/TestRatMatData/TestRAREImage2.mat'
    # realTestdata1 = sio.loadmat(realTestpath1)
    # x_test = realTestdata1['ImagData']
    # realTestdata2 = sio.loadmat(realTestpath2)
    # y_test = realTestdata2['ImagData']


    # x_test_1 = realTestdata[:,:,0,:]
    # x_test_2 = realTestdata[:,:,1,:]
    # x_test = np.concatenate((x_test_1,x_test_2),axis=2)
    # x_test = moveaxis(moveaxis(x_test,0,2),0,1)


    for i in range(x_test.shape[0]):
            x_test_3c[i, :, :, :] = cv2.normalize(x_test_3c[i, :, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
            y_test_3c[i, :, :, :] = cv2.normalize(y_test_3c[i, :, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)

    # x_test = np.reshape(x_test, (x_test.shape[0], 256, 256, 1))
    # y_test = np.reshape(y_test, (y_test.shape[0], 256, 256, 1))
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

    generated_images = g.predict(x=x_test_3c)

    # np.save('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/test_motion_image.npy',
    #         x_test)
    # np.save('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/test_gt_image.npy',
    #         y_test)
    # generated_images = np.array(generated_images)
    # print('shape',generated_images.shape)
    # generated = np.array([deprocess_image(img) for img in generated_images])
    generated = np.array(generated_images)
    np.save(os.path.join(output_dir_1, 'motion_test_Channel3_1.npy'), x_test_3c)
    np.save(os.path.join(output_dir_1, 'gt_test_Channel3_1.npy'), y_test_3c)
    np.save(os.path.join(output_dir_1, 'pred_test_Channel3_1.npy'), generated)
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

        dimg1 = generated[i, :, :, :]
        dimg1 = dimg1.astype("float64")
        dimg2 = y_test_3c[i, :, :, :]
        dimg3 = x_test_3c[i, :, :, :]
        ssim = compare_ssim(dimg1, dimg2, multichannel=True)
        psnr = compare_psnr(dimg1, dimg2)
        mse = compare_mse(dimg1, dimg2)
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
        # plt.savefig('/home/user/data1/yalei/mocoGAN/moco-gan-master/scripts/test_0916_channel3/%s_%d_%s.png' % ('channel3', i,"motion_fse"), bbox_inches='tight', dpi=400,
        #             pad_inches=0)

        # dimg  = np.concatenate((dimg1,dimg2,dimg3),axis=1)
        dimg  = np.concatenate((dimg2,dimg1,dimg3),axis=1)
        plt.figure(num=2)
        plt.imshow(dimg,cmap='gray')
        plt.axis("off")

        # plt.show()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.savefig('/home/user/data1/yalei/mocoGAN/moco-gan-master/scripts/test_0818_deepnet/%s_%d_%s.png' % ('test', i, "real_motion_fse"), bbox_inches='tight', dpi=400,
        #             pad_inches=0)
        plt.savefig('E:/chenyalei/mocogan_v2/scripts/test_pre_l1_simu_0927/%s_%d_%s.png' % ('channel3', i,"motion_fse"), bbox_inches='tight', dpi=400,
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
@click.option('--weight_path', default='E:/chenyalei/mocogan_v2/scripts/weights/927/generator_89_40.h5',help='Model weight')
@click.option('--input_dir', default='E:/chenyalei/deblur-gan-master/test_motion_424.mat',help='Image to deblur')
@click.option('--output_dir', default='E:/chenyalei/deblur-gan-master/images/output427',help='Deblurred image')
def deblur_command(weight_path, input_dir, output_dir):

    return deblur(weight_path, input_dir, output_dir)

# weight_path = 'E:/chenyalei/deblur-gan-master/weights/1116/generator_99_54.h5'
# input_dir = 'E:/chenyalei/deblur-gan-master/images/test/A'
# output_dir = 'E:/chenyalei/deblur-gan-master/images/output1116'


if __name__ == "__main__":
	deblur_command()
