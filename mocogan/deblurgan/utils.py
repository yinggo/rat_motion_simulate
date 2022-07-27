import os
from PIL import Image
import numpy as np
import tensorflow as tf
# import cv2
import random
from keras.callbacks import TensorBoard
from keras.callbacks import TensorBoard
from skimage.transform import resize
from scipy import io
import scipy.io as sio
import keras.backend as K
from keras.layers import Dense,GlobalAveragePooling2D,Reshape,Add,GlobalMaxPooling2D,Activation,Permute,Lambda,Concatenate,Conv2D
from numpy import moveaxis,transpose,multiply

RESHAPE = (256,256)

# load ground truth, training, and test data
# data_fn = 'E:/chenyalei/deblur-gan-master/dataset/traindata.mat'
# tmp_mat_data = sio.loadmat(data_fn)
# x_train = np.concatenate((tmp_mat_data['x_train_pt1'],tmp_mat_data['x_train_pt2']), axis=0)
# y_train = np.concatenate((tmp_mat_data['y_train_pt1'],tmp_mat_data['y_train_pt2']), axis=0)
# x_test = tmp_mat_data['x_test']
# y_test = tmp_mat_data['y_test']
# print("Data structures loaded.")

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def loadmat_motion(input_dir,input_size_1 = 256,input_size_2 = 256):
    mat= sio.loadmat(input_dir)
    # imgs_A1 = mat['real_data']
    # imgs_A2 = mat['imag_data']
    imgs_train = mat['motiondata']
    # x_train = np.zeros(shape=(1410, 256, 256, 2))
    # x_train[:, :, :, 0] = imgs_A1
    # x_train[:, :, :, 1] = imgs_A2
    # imgs_train = moveaxis(imgs_train, 1, 2)
    imgs_train = abs(imgs_train)
    imgs_train = resize(imgs_train, [imgs_train.shape[0],input_size_1, input_size_2])
    imgs_train = np.reshape(imgs_train,[imgs_train.shape[0],imgs_train.shape[1],imgs_train.shape[2],1])

    return imgs_train

def loadmat_gt(input_dir,input_size_1 = 256,input_size_2 = 256):
    mat= sio.loadmat(input_dir)
    # imgs_A1 = mat['real_data']
    # imgs_A2 = mat['imag_data']
    imgs_train = mat['GTdata']
    # x_train = np.zeros(shape=(1410, 256, 256, 2))
    # x_train[:, :, :, 0] = imgs_A1
    # x_train[:, :, :, 1] = imgs_A2
    # imgs_train = moveaxis(imgs_train, 1, 2)
    imgs_train = abs(imgs_train)
    imgs_train = resize(imgs_train, [imgs_train.shape[0],input_size_1, input_size_2])
    imgs_train = np.reshape(imgs_train, [imgs_train.shape[0], imgs_train.shape[1], imgs_train.shape[2], 1])
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
    # imgs_train = transpose(imgs_train)
    imgs_train = abs(imgs_train)
    imgs_train = resize(imgs_train, [imgs_train.shape[0],input_size_1, input_size_2])
    # imgs_train = np.reshape(imgs_train, [imgs_train.shape[0], imgs_train.shape[1], imgs_train.shape[2], 1])
    # for i in range(imgs_train.shape[0]):
    #     imgs_train[i,:,:,0] = (imgs_train[i,:,:,0] - 127.5)/127.5
    #     # img = (img - 127.5) / 127.5
    return imgs_train

def loadmat_mouse(input_dir,input_size_1 = 256,input_size_2 = 256):
    mat= sio.loadmat(input_dir)
    # imgs_A1 = mat['real_data']
    # imgs_A2 = mat['imag_data']
    imgs_train = mat['motionData']
    # x_train = np.zeros(shape=(1410, 256, 256, 2))
    # x_train[:, :, :, 0] = imgs_A1
    # x_train[:, :, :, 1] = imgs_A2
    # imgs_train = moveaxis(imgs_train, 1, 2)
    # imgs_train = transpose(imgs_train)
    imgs_train = abs(imgs_train)
    # imgs_train = resize(imgs_train, [imgs_train.shape[0],input_size_1, input_size_2])
    # imgs_train = np.reshape(imgs_train, [imgs_train.shape[0], imgs_train.shape[1], imgs_train.shape[2], 1])
    # for i in range(imgs_train.shape[0]):
    #     imgs_train[i,:,:,0] = (imgs_train[i,:,:,0] - 127.5)/127.5
    #     # img = (img - 127.5) / 127.5
    return imgs_train

# def random_crop(image, crop_shape):
#     image = image[0,:,:,0]
#     if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
#         x = random.randrange(image.shape[1] - crop_shape[0])
#         y = random.randrange(image.shape[0] - crop_shape[1])
#         image = image[y:y + crop_shape[1], x:x + crop_shape[0]]
#         image = np.reshape(image,(1,image.shape[0],image.shape[1],1))
#         return image
#     else:
#         image = cv2.resize(image, crop_shape)
#         return image

def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]



# def image_concat(divide_image):
#     # m,n,grid_h, grid_w=[divide_image.shape[0],divide_image.shape[1],#每行，每列的图像块数
#     #                    divide_image.shape[2],divide_image.shape[3]]#每个图像块的尺寸
#     m = 4
#     n = 4
#     grid_h = 64
#     grid_w = 64
#     restore_image = np.zeros([m*grid_h, n*grid_w, 1], np.uint8)
#     restore_image[0:grid_h,0:]
#     for i in range(m):
#         for j in range(n):
#             restore_image[i*grid_h:(i+1)*grid_h,j*grid_w:(j+1)*grid_w]=divide_image[i,:,:,:]
#     return restore_image


# def divide_method2(img, m, n):  # 分割成m行n列
#     h, w = img.shape[0], img.shape[1]
#     grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
#     grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽
#
#     # 满足整除关系时的高、宽
#     h = grid_h * (m - 1)
#     w = grid_w * (n - 1)
#
#     # 图像缩放
#     # img_re = cv2.resize(img, (w, h),
#     #                     cv2.INTER_LINEAR)  # 也可以用img_re=skimage.transform.resize(img, (h,w)).astype(np.uint8)
#     # plt.imshow(img_re)
#     gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
#     gx = gx.astype(np.int)
#     gy = gy.astype(np.int)
#
#     divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 1],
#                             np.uint8)
#     divide_image_sum = np.zeros([4,grid_h,grid_w,1],np.uint8)# 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
#     # divide_image = np.zeros([ grid_h, grid_w, 1],np.uint8)
#     for i in range(m - 1):
#         for j in range(n - 1):
#             divide_image[i, j, ...] = img[ gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]
#             divide_image_sum[j,:,:,:] = divide_image[i,j,:,:,:]
#     return divide_image_sum
#
#     img_size = int(path_size * scale)
#     I = cv2.imread(img)
#     I = random_crop(I, (img_size, img_size))
#     y = I.copy()
#     # Use different downsampling methods
#     if np.random.randint(2): # x_scale sampling
#         I = I[::scale, ::scale]
#     else: #bilinear resizing
#         I = cv2.resize(I, (path_size, path_size))
#     return I, y

def load_image(path):
    img = Image.open(path)
    return img


def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    # A_paths = 'E:/chenyalei/deblur-gan-master/images/train/A'
    # B_paths = 'E:/chenyalei/deblur-gan-master/images/train/B'
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }
# def load_mat(path, n_images):
#     if n_images < 0:
#         n_images = float("inf")
#     # A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
#     A_path = 'E:/chenyalei/deblur-gan-master/dataset/traindata.mat'
#     B_path = 'E:/chenyalei/deblur-gan-master/dataset/traingt.mat'
#     mat_A = sio.loadmat(A_path)
#     mat_B = sio.loadmat(B_path)
#     imgs_A = mat_A['H']
#     imgs_B = mat_B['H']
#     # all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
#     images_A, images_B = [], []
#     images_A_paths, images_B_paths = [], []
#     for i, j in zip(imgs_A, imgs_B):
#         img_A = imgs_A[i,:,:]
#         img_B = imgs_B[j,:,:]
#         images_A.append(img_A)
#         images_B.append(img_B)
#         # images_A_paths.append(path_A)
#         # images_B_paths.append(path_B)
#         if len(images_A) > n_images - 1: break
#
#     return {
#         'A': np.array(images_A),
#         # 'A_paths': np.array(images_A_paths),
#         'B': np.array(images_B),
#         # 'B_paths': np.array(images_B_paths)
#     }

def write_log(callback, names, logs, batch_no):
    """
    Util to write callback for Keras training
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        # writer = tf.summary.FileWriter(logs)
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def add_noisy(input_img, param=25):
    # param = 25
    # 灰阶范围
    grayscale = 256
    img = input_img
    w = img.shape[1]
    h = img.shape[0]
    newimg = np.zeros(h, w)

    for x in range(0, h):
        for y in range(0, w, 2):
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()
            z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
            z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))

            fxy = int(img[x, y] + z1)
            fxy1 = int(img[x, y + 1] + z2)
            # f(x,y)
            if fxy < 0:
                fxy_val = 0
            elif fxy > grayscale - 1:
                fxy_val = grayscale - 1
            else:
                fxy_val = fxy
            # f(x,y+1)
            if fxy1 < 0:
                fxy1_val = 0
            elif fxy1 > grayscale - 1:
                fxy1_val = grayscale - 1
            else:
                fxy1_val = fxy1
            newimg[x, y] = fxy_val
            newimg[x, y + 1] = fxy1_val
    return newimg

def gasuss_noise(image, mean=0, var=0.05):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    # noise_img = np.zeros(shape=(image.shape[0],image.shape[1],image.shape[2],image.shape[3]))
    # for i in range(image.shape[0]):
    # image = image[i,:,:]
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    # noise_img[i,:,:,:] = out
    # out = np.reshape(out,(1,256,256,1))
    # if out.min() < 0:
    #     low_clip = -1.
    # else:
    #     low_clip = 0.
    # out = np.clip(out, low_clip, 1.0)
    # # out = np.uint8(out*255)
    # out = np.reshape(out,(image.shape[0],image.shape[1],image.shape[2],1))
    # cv2.imshow("gasuss", out)
    return out
# class Mylosscallback(Callback):
#     def __init__(self, log_dir):
#         super().__init__()
#         self.val_writer = tf.summary.FileWriter(log_dir)
#         self.num = 0
#
#     def on_train_begin(self, logs={}):
#         self.losses = []
#
#     def on_batch_end(self, batch, logs={}):
#         self.num = self.num + 1
#         val_loss = logs.get('loss')
#         # print(1111)
#         val_loss_summary = tf.Summary()
#         val_loss_summary_value = val_loss_summary.value.add()
#         val_loss_summary_value.simple_value = val_loss
#         val_loss_summary_value.tag = 'loss'
#         self.val_writer.add_summary(val_loss_summary, self.num)
#         self.val_writer.flush()
def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    ch_weight = Lambda(lambda x:x*cbam_feature)
    output = ch_weight(input_feature)
    return output


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    sp_weight = Lambda(lambda x:x*cbam_feature)
    output = sp_weight(input_feature)
    return output



def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

