import numpy as np
import random
import os
import scipy.io as sio
import h5py
import datetime
from keras.models import save_model, load_model, Model
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Activation, Subtract
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from numpy import moveaxis
import pickle
import cv2
import tqdm
# input_name = os.listdir('train_data3/JPEGImages')

# n = len(input_name)
BASE_DIR = 'E:/chenyalei/mocogan_v2/scripts/unet_weight'

def save_all_weights(model, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save_weights(os.path.join(save_dir, 'unet_{}_{}.h5'.format(epoch_number, current_loss)), True)

# n =  1410
# batch_size = 1
input_size_1 = 256
input_size_2 = 256
def loadmat_gt(input_dir):
    mat= sio.loadmat(input_dir)
    # imgs_A1 = mat['real_data']
    # imgs_A2 = mat['imag_data']
    imgs_train = mat['trainGTdata']
    # x_train = np.zeros(shape=(1410, 256, 256, 2))
    # x_train[:, :, :, 0] = imgs_A1
    # x_train[:, :, :, 1] = imgs_A2
    imgs_train = moveaxis(imgs_train, 1, 2)
    return imgs_train
def loadmat_motion(input_dir):
    mat= sio.loadmat(input_dir)
    # imgs_A1 = mat['real_data']
    # imgs_A2 = mat['imag_data']
    imgs_train = mat['trainMotiondata']
    # x_train = np.zeros(shape=(1410, 256, 256, 2))
    # x_train[:, :, :, 0] = imgs_A1
    # x_train[:, :, :, 1] = imgs_A2
    imgs_train = moveaxis(imgs_train, 1, 2)
    imgs_train = abs(imgs_train)
    return imgs_train

def loadmat_c(input_dir,input_size_1 = 256,input_size_2 = 256):
    mat= sio.loadmat(input_dir)
    # imgs_A1 = mat['real_data']
    # imgs_A2 = mat['imag_data']
    imgs_train = mat['data']
    # x_train = np.zeros(shape=(1410, 256, 256, 2))
    # x_train[:, :, :, 0] = imgs_A1
    # x_train[:, :, :, 1] = imgs_A2
    # imgs_train = moveaxis(imgs_train, 1, 2)
    imgs_train = abs(imgs_train)
    imgs_train = resize(imgs_train, [imgs_train.shape[0],input_size_1, input_size_2])
    imgs_train = np.reshape(imgs_train, [imgs_train.shape[0], imgs_train.shape[1], imgs_train.shape[2], 1])
    return imgs_train


def loadmat_h5py(input_dir):
    mat= h5py.File(input_dir)
    imgs_A1 = mat['real_data']
    imgs_A2 = mat['imag_data']
    x_train = np.zeros(shape=(256, 256, 130, 2))
    x_train[:, :, :, 0] = imgs_A1
    x_train[:, :, :, 1] = imgs_A2
    imgs_train = moveaxis(moveaxis(x_train, 2, 0), 1, 2)
    return imgs_train

def batch_data(input_x,input_y,n, batch_size, input_size_1=256, input_size_2=256):
    rand_num = random.randint(0,n)
    # img1 = io.imread('train_data3/JPEGImages/' + input_name[rand_num]).astype("float")
    # img2 = io.imread('train_data3/TargetImages/' + input_name[rand_num]).astype("float")
    # img1 = loadmat_c('E:/chenyalei/DATASET single/HCP_motion_c_single/HCP_motion_c_single1.mat')
    # img2 = loadmat_c('E:/chenyalei/DATASET single/HCP_GT_c_single/HCP_GT_c_single1.mat')
    # img1 = loadmat_c('E:/chenyalei/DATASET single/HCP_motion_5_single_1-3.mat')
    # img2 = loadmat_c('E:/chenyalei/DATASET single/HCP_GT_5/HCP_GT_5_single_1-3.mat')
    img1 = input_x[rand_num,:,:]
    img2 = input_y[rand_num,:,:]
    img1 = resize(img1, [input_size_1, input_size_2, 1])
    img2 = resize(img2, [input_size_1, input_size_2, 1])

    img1 = cv2.normalize(img1, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
    img2 = cv2.normalize(img2, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)

    img1 = np.reshape(img1, (1, input_size_1, input_size_2, 1))
    img2 = np.reshape(img2, (1, input_size_1, input_size_2, 1))

    batch_input = img1
    batch_output = img2
    for batch_iter in range(1, batch_size):
        rand_num = random.randint(0,1799)
        # img1 = io.imread('train_data3/JPEGImages/' + input_name[rand_num]).astype("float")
        # img2 = io.imread('train_data3/TargetImages/' + input_name[rand_num]).astype("float")
        # img1 = loadmat_c('E:/chenyalei/DATASET single/HCP_motion_5_single_1-3.mat')
        # img2 = loadmat_c('E:/chenyalei/DATASET single/HCP_GT_5/HCP_GT_5_single_1-3.mat')
        img1 = img1[rand_num, :, :]
        img2 = img2[rand_num, :, :]
        img1 = resize(img1, [input_size_1, input_size_2, 1])
        img2 = resize(img2, [input_size_1, input_size_2, 1])

        img1 = cv2.normalize(img1, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
        img2 = cv2.normalize(img2, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
        img1 = np.reshape(img1, (1, input_size_1, input_size_2, 1))
        img2 = np.reshape(img2, (1, input_size_1, input_size_2, 1))

        batch_input = np.concatenate((batch_input, img1), axis=0)
        batch_output = np.concatenate((batch_output, img2), axis=0)
    return batch_input, batch_output


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def random_crop(image, crop_shape):
    image = image[0,:,:,0]
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1] - crop_shape[0])
        y = random.randrange(image.shape[0] - crop_shape[1])
        image = image[y:y + crop_shape[1], x:x + crop_shape[0]]
        image = np.reshape(image,(1,image.shape[0],image.shape[1],1))
        return image
    else:
        image = cv2.resize(image, crop_shape)
        return image


def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

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
dncnn = DnCNN()
dncnn.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mse'])
dncnn.summary()


# inpt = Input(shape=(input_size_1, input_size_2, 1))
#
# conv1 = Conv2d_BN(inpt, 64, (3, 3))
# conv1 = Conv2d_BN(conv1,64, (3, 3))
# pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
#
# conv2 = Conv2d_BN(pool1, 128, (3, 3))
# conv2 = Conv2d_BN(conv2, 128, (3, 3))
# pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
#
# conv3 = Conv2d_BN(pool2, 256, (3, 3))
# conv3 = Conv2d_BN(conv3, 256, (3, 3))
# pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
#
# conv4 = Conv2d_BN(pool3, 512, (3, 3))
# conv4 = Conv2d_BN(conv4, 512, (3, 3))
# pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
#
# conv5 = Conv2d_BN(pool4, 1024, (3, 3))
# conv5 = Dropout(0.5)(conv5)
# conv5 = Conv2d_BN(conv5, 512, (3, 3))
# conv5 = Dropout(0.5)(conv5)
#
# convt1 = Conv2dT_BN(conv5, 512, (3, 3))
# concat1 = concatenate([conv4, convt1], axis=3)
# concat1 = Dropout(0.5)(concat1)
# conv6 = Conv2d_BN(concat1, 512, (3, 3))
# conv6 = Conv2d_BN(conv6, 512, (3, 3))
#
# convt2 = Conv2dT_BN(conv6, 256, (3, 3))
# concat2 = concatenate([conv3, convt2], axis=3)
# concat2 = Dropout(0.5)(concat2)
# conv7 = Conv2d_BN(concat2, 256, (3, 3))
# conv7 = Conv2d_BN(conv7, 256, (3, 3))
#
# convt3 = Conv2dT_BN(conv7, 128, (3, 3))
# concat3 = concatenate([conv2, convt3], axis=3)
# concat3 = Dropout(0.5)(concat3)
# conv8 = Conv2d_BN(concat3, 128, (3, 3))
# conv8 = Conv2d_BN(conv8, 128, (3, 3))
#
# convt4 = Conv2dT_BN(conv8, 64, (3, 3))
# concat4 = concatenate([conv1, convt4], axis=3)
# concat4 = Dropout(0.5)(concat4)
# conv9 = Conv2d_BN(concat4, 64, (3, 3))
# conv9 = Conv2d_BN(conv9, 64, (3, 3))
# conv9 = Dropout(0.5)(conv9)
# outpt = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv9)
#
# model = Model(inpt, outpt)
# model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mse'])
# model.summary()
# rootdir1 = 'E:/chenyalei/DATASET single/HCP_motion_5_single_1-3.mat'
# rootdir2 = 'E:/chenyalei/DATASET single/HCP_GT_5/HCP_GT_5_single_1-3.mat'
# x_train_all = sio.loadmat(rootdir1)
# y_train_all = sio.loadmat(rootdir2)
# x_train_all = x_train_all['data']
# y_train_all = y_train_all['data']
# # rootdir1 = '/home/user/data1/mouse_data/FullBrain/train/gt/'
# # rootdir2 = '/home/user/data1/mouse_data/FullBrain/train/motion/'
# list1 = os.listdir(rootdir1)
# list2 = os.listdir(rootdir2)
# list1.sort(key=lambda x: int(x[24:-4]))
# list2.sort(key=lambda x: int(x[20:-4]))
#
# y_train_1 = np.zeros(shape=(1, 256, 256), dtype='float64')
# x_train_1 = np.zeros(shape=(1, 256, 256), dtype='float64')
#
# for num in range(0, 150):
#     path1 = os.path.join(rootdir1, list1[num])
#     motion_img = sio.loadmat(path1)
#     motion_img = abs(motion_img['data1'])
#
#     x_train_1 = np.concatenate((x_train_1, motion_img), axis=0)
#
#     path2 = os.path.join(rootdir2, list2[num])
#     GT1 = sio.loadmat(path2)
#     GT1 = GT1['data2']
#     y_train_1 = np.concatenate((y_train_1, GT1), axis=0)
#
# x_train_all = x_train_1[1:, :, :]
# y_train_all = y_train_1[1:, :, :]
x_train_1 = np.load('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/hcp_motion_1-3.npy')
y_train_1 = np.load('E:/chenyalei/deblur-gan-master/dataset/GT_data_611/hcp_GT_1-3.npy')
x_train_2 = np.load('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/hcp_motion_4-6.npy')
y_train_2 = np.load('E:/chenyalei/deblur-gan-master/dataset/GT_data_611/hcp_GT_4-6.npy')
x_train_3 = np.load('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/hcp_motion_7-9.npy')
y_train_3 = np.load('E:/chenyalei/deblur-gan-master/dataset/GT_data_611/hcp_GT_7-9.npy')

# x_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/knee_data/knee_motion/knee_motion_629.npy')
# y_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/knee_data/knee_gt/knee_gt_629.npy')
x_train_4 = np.load('E:/chenyalei/DATASET single/motionnum16/hcp_motion_13_17.npy')
y_train_4 = np.load('E:/chenyalei/DATASET single/HCP_GT_13-17/hcp_GT_13_17.npy')
x_train_all = np.concatenate((x_train_1, x_train_2, x_train_3, x_train_4), axis=0)
y_train_all = np.concatenate((y_train_1, y_train_2, y_train_3, y_train_4), axis=0)

for i in range(x_train_all.shape[0]):
    x_train_all[i, :, :] = cv2.normalize(x_train_all[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
    y_train_all[i, :, :] = cv2.normalize(y_train_all[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)

x_train_all = np.reshape(x_train_all,(x_train_all.shape[0],x_train_all.shape[1],x_train_all.shape[2],1))
y_train_all = np.reshape(y_train_all,(y_train_all.shape[0],y_train_all.shape[1],y_train_all.shape[2],1))

epoch_num = 100
batch_size = 4
for epoch in tqdm.tqdm(range(epoch_num)):
    permutated_indexes = np.random.permutation(x_train_all.shape[0])

    unet_losses = []
    # d_losses_fake = []
    # d_losses_real = []
    # d_on_g_losses = []
    # test_losses = []
    for index in range(int(x_train_all.shape[0] / batch_size)):
        batch_indexes = permutated_indexes[index * batch_size:(index + 1) * batch_size]
        image_blur_batch = x_train_all[batch_indexes]
        image_clean_batch = y_train_all[batch_indexes]

        unet_loss = dncnn.train_on_batch(image_blur_batch, image_clean_batch)
        unet_losses.append(unet_loss)

    print(np.mean(unet_losses))

    with open('log_1012_dncnn_epoch100.txt', 'a+') as f:
        f.write('{} - {} \n'.format(epoch, np.mean(unet_losses)))

    save_all_weights(dncnn, epoch, int(np.mean(unet_losses)))
# def DnCNN(depth, filters=64, image_channels=1, use_bnorm=True):
#     layer_count = 0
#     inpt = Input(shape=(None, None, image_channels), name='input' + str(layer_count))
#     # 1st layer, Conv+relu
#     layer_count += 1
#     x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
#                name='conv' + str(layer_count))(inpt)
#     layer_count += 1
#     x = Activation('relu', name='relu' + str(layer_count))(x)
#     # depth-2 layers, Conv+BN+relu
#     for i in range(depth - 2):
#         layer_count += 1
#         x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
#                    use_bias=False, name='conv' + str(layer_count))(x)
#         if use_bnorm:
#             layer_count += 1
#             # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
#             x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn' + str(layer_count))(x)
#         layer_count += 1
#         x = Activation('relu', name='relu' + str(layer_count))(x)
#         # last layer, Conv
#     layer_count += 1
#     x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
#                padding='same', use_bias=False, name='conv' + str(layer_count))(x)
#     layer_count += 1
#     x = Subtract(name='subtract' + str(layer_count))([inpt, x])  # input - noise
#     model = Model(inputs=inpt, outputs=x, name='DnCNN')
#
#     return model
#
# model = DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True)
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])

#
# itr = 700
# S = []
# for i in range(itr):
#     print("iteration = ", i + 1)
#     if i < 500:
#         bs = 4
#     elif i < 2000:
#         bs = 8
#     elif i < 5000:
#         bs = 16
#     else:
#         bs = 32
# # img1 = loadmat('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_trainMotiondata_moco1.mat')
# # img2 = loadmat('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_trainGTdata_moco1.mat')
#     x_train_batch, y_train_batch = batch_data(x_train_all.shape[0], x_train_all,y_train_all,batch_size=16)
#     # train_X = random_crop(train_X,(128,128))
#     # train_Y = random_crop(train_Y,(128,128))
# # train_X, train_Y = img1,img2
#     history = model.fit(x_train_batch, y_train_batch, validation_split=0.2 , epochs=1, verbose=1)
# #
# # save_model(model,'unet423_1.h5')
#     if i % 100 == 0:
#         save_model(model, 'unet_0812_{}.h5'.format(i))
#
# with open('trainHistoryDict_0812.txt', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
#
# net_architecture = 'unet_0812'
# # model = load_model('unet423_2.h5')
# history_loss = history.history['loss']
# history_val_loss = history.history['val_loss']
# # axloss.plot(history_loss)
# np_loss = np.array(history_loss)
# np_val_loss = np.array(history_val_loss)
# filePath = '/home/user/data1/yalei/mocoGAN/moco-gan-master/scripts/unet_history'
# filename_loss = filePath + net_architecture + '_loss.txt'
# filename_val_loss = filePath + net_architecture + '_val_loss.txt'
# np.savetxt(filename_loss,np_loss)
# np.savetxt(filename_val_loss,np_val_loss)
# model = load_model('unet423_2.h5')


# def batch_data_test(n, batch_size=8, input_size_1=256, input_size_2=256):
#     rand_num = random.randint(0, 129)
#     # img1 = io.imread('test_data3/JPEGImages/' + input_name[rand_num]).astype("float")
#     # img2 = io.imread('test_data3/TargetImages/' + input_name[rand_num]).astype("float")
#     img1 = loadmat_h5py('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_testMotionData.mat')
#     img2 = loadmat_h5py('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_testGTData.mat')
#     img1 = img1[rand_num,:,:,:]
#     img2 = img2[rand_num,:,:,:]
#     img1 = resize(img1, [input_size_1, input_size_2, 2])
#     img2 = resize(img2, [input_size_1, input_size_2, 2])
#     img1 = np.reshape(img1, (1, input_size_1, input_size_2, 2))
#     img2 = np.reshape(img2, (1, input_size_1, input_size_2, 2))
#     img1 /= 255
#     img2 /= 255
#     batch_input = img1
#     batch_output = img2
#     for batch_iter in range(1, batch_size):
#         rand_num = random.randint(0, 129)
#         # img1 = io.imread('test_data3/JPEGImages/' + input_name[rand_num]).astype("float")
#         # img2 = io.imread('test_data3/TargetImages/' + input_name[rand_num]).astype("float")
#         img1 = loadmat_h5py('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_testMotionData.mat')
#         img2 = loadmat_h5py('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_testGTData.mat')
#         img1 = img1[rand_num, :, :, :]
#         img2 = img2[rand_num, :, :, :]
#         img1 = resize(img1, [input_size_1, input_size_2, 2])
#         img2 = resize(img2, [input_size_1, input_size_2, 2])
#         img1 = np.reshape(img1, (1, input_size_1, input_size_2, 2))
#         img2 = np.reshape(img2, (1, input_size_1, input_size_2, 2))
#         img1 /= 255
#         img2 /= 255
#         batch_input = np.concatenate((batch_input, img1), axis=0)
#         batch_output = np.concatenate((batch_output, img2), axis=0)
#     return batch_input, batch_output


# test_name = os.listdir('test_data3/JPEGImages')
# n_test = len(test_name)
# test_img1 = loadmat_h5py('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_testMotionData.mat')
# test_img2 = loadmat_h5py('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_testGTData.mat')
# test_X, test_Y = batch_data_test(batch_size=10)
# # test_X, test_Y = test_img1,test_img2
# pred_Y = model.predict(test_X)
# ii = 0
# plt.figure()
# plt.imshow(test_X[ii, :, :, 0])
# plt.axis('off')
# plt.figure()
# plt.imshow(test_Y[ii, :, :, 0])
# plt.axis('off')
# plt.figure()
# plt.imshow(pred_Y[ii, :, :, 0])
# plt.axis('off')
