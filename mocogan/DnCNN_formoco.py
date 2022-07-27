import numpy as np
import keras.backend as K
import random
import os
import scipy.io as sio
import h5py
from keras.models import save_model, load_model, Model
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate,Add,UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Activation, Subtract
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from numpy import moveaxis
import pickle
from skimage.measure import compare_ssim
from keras import backend as K
import tensorflow as tf
# input_name = os.listdir('train_data3/JPEGImages')

# n = len(input_name)
n =  1410
batch_size = 1
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

def batch_data(n, batch_size=8, input_size_1=256, input_size_2=256):
    rand_num = random.randint(0, 1799)
    # img1 = io.imread('train_data3/JPEGImages/' + input_name[rand_num]).astype("float")
    # img2 = io.imread('train_data3/TargetImages/' + input_name[rand_num]).astype("float")
    img1 = loadmat_c('E:/chenyalei/DATASET single/HCP_motion_5_single_1-3.mat')
    img2 = loadmat_c('E:/chenyalei/DATASET single/HCP_GT_5/HCP_GT_5_single_1-3.mat')
    img1 = img1[rand_num,:,:]
    img2 = img2[rand_num,:,:]
    img1 = resize(img1, [input_size_1, input_size_2, 1])
    img2 = resize(img2, [input_size_1, input_size_2, 1])
    img1 = np.reshape(img1, (1, input_size_1, input_size_2, 1))
    img2 = np.reshape(img2, (1, input_size_1, input_size_2, 1))
    img1 /= 255
    img2 /= 255
    batch_input = img1
    # batch_output = img2
    batch_output = img1-img2
    for batch_iter in range(1, batch_size):
        rand_num = random.randint(0,1799)
        # img1 = io.imread('train_data3/JPEGImages/' + input_name[rand_num]).astype("float")
        # img2 = io.imread('train_data3/TargetImages/' + input_name[rand_num]).astype("float")
        img1 = loadmat_c('E:/chenyalei/DATASET single/HCP_motion_5_single_1-3.mat')
        img2 = loadmat_c('E:/chenyalei/DATASET single/HCP_GT_5/HCP_GT_5_single_1-3.mat')
        img1 = img1[rand_num, :, :]
        img2 = img2[rand_num, :, :]
        img1 = resize(img1, [input_size_1, input_size_2, 1])
        img2 = resize(img2, [input_size_1, input_size_2, 1])
        img1 = np.reshape(img1, (1, input_size_1, input_size_2, 1))
        img2 = np.reshape(img2, (1, input_size_1, input_size_2, 1))
        img1 /= 255
        img2 /= 255
        batch_input = np.concatenate((batch_input, img1), axis=0)
        batch_output = np.concatenate((batch_output,  img1-img2), axis=0)
    return batch_input, batch_output


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same',kernel_mode = 'he_normal'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding,kernel_mode =kernel_mode )(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def conv2d_bn(x, nb_filter, kernel_size, stride, border_mode='same',kernel_mode = 'he_normal',activ_fn='Relu',bias=False):
    x = Conv2D(nb_filter, kernel_size, strides=stride, padding=border_mode )(x)
    x = BatchNormalization(axis=3)(x)

    x = Activation(activ_fn)(x)
    return x

def conv2d_bn_drop(x, nb_filter, kernel_size, stride, border_mode='same',kernel_mode = 'he_normal',activ_fn='Relu',bias=False):
    x = Conv2D(nb_filter, kernel_size, strides=stride, padding=border_mode )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.2)(x)
    x = Activation(activ_fn)(x)
    return x

def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
def reduction_block(input,kshape,kshape2,filter2):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input._keras_shape[channel_axis]
    r1 =conv2d_bn(input,filters , kshape2, stride=1, activ_fn='relu', border_mode='same'
                    , bias=False)
    r2 =conv2d_bn(r1,filters , kshape, stride=1, activ_fn='relu', border_mode='same',
                       bias=False)
    r3 = conv2d_bn(r2, filter2, kshape2, stride=1, activ_fn='relu', border_mode='same',
                   bias=False)
    return r3
def inception_resnet(input0,kshape,kshape2,filter2,subsample=True):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input0._keras_shape[channel_axis]
    input=Activation('relu')(input0)
    i1 = conv2d_bn(input, filter2, kshape2, stride=1, activ_fn='relu', border_mode='same',
                   bias=False)
    i2 = conv2d_bn(input, filter2, kshape2, stride=1, activ_fn='relu', border_mode='same',
                    bias=False)
    i2 = conv2d_bn(i2, filter2, kshape, stride=1, activ_fn='relu', border_mode='same',
                    bias=False)
    i3 = conv2d_bn(input, filter2, kshape, stride=1, activ_fn='relu', border_mode='same',
                    bias=False)
    i3 = conv2d_bn(i3, filter2*1.5, kshape, stride=1, activ_fn='relu', border_mode='same',
                    bias=False)
    i3 = conv2d_bn(i3, filter2*2, kshape, stride=1, activ_fn='relu', border_mode='same',
                    bias=False)
    c1 =concatenate([i1,i2,i3], axis=-1)
    print(c1.shape)
    c1 =conv2d_bn(c1, filters, kshape2, stride=1, activ_fn='linear', border_mode='same',
                    bias=False,normalize=False)
    c1 =Add()([c1, input])
    c1=Activation('relu')(c1)
    if subsample:
        c2= MaxPooling2D()(c1)
    else: c2=c1
    return c2

def resnetv2_for_motion_normalsize(H=256, W=256, channels=1,  kshape=(3, 3),kshape2=(1, 1),kshape3=(5, 5),reduction=True):
    nb_filter = 32
    inputs = Input(shape=(H, W, channels))
    inputs = tf.cast(inputs, tf.int32)
    conv1 = conv2d_bn(inputs,nb_filter,kshape,stride=2, activ_fn='relu', border_mode='same',bias=False)
    print(conv1.shape)
    conv2 = conv2d_bn(conv1, nb_filter*2, kshape, stride=1, activ_fn='relu', border_mode='same',
                      bias=False)
    print(conv2.shape)
    pool1 =MaxPooling2D(pool_size=(3,3),strides=2)(conv2)
    print(pool1.shape)

    conv3 = conv2d_bn(pool1, 80, kshape, stride=1, activ_fn='relu', border_mode='same',
                     bias=False)
    print(conv3.shape)
    conv4 = conv2d_bn(conv3, nb_filter * 6, kshape, stride=1, activ_fn='relu', border_mode='same',
                      bias=False)
    print(conv4.shape)
    pool2 = MaxPooling2D(pool_size=(4, 4), strides=2)(conv4)
    print(pool2.shape)
    if reduction:
        conv5 = reduction_block(pool2,kshape,kshape2,nb_filter * 10)
    else:
        conv5 = conv2d_bn(pool2, nb_filter * 8, kshape, stride=1, activ_fn='relu', border_mode='same',
                bias=False)
    print(conv5.shape)
    conv5_1=conv5
    for i in range(9):
        conv5_1 =inception_resnet(conv5_1, kshape, kshape2, nb_filter , subsample=False)
    conv6 = inception_resnet(conv5_1, kshape, kshape2, nb_filter, subsample=True)
    print(conv6.shape)
    if reduction:
        conv7 = reduction_block(conv6,kshape,kshape2,nb_filter * 34)
    else:
        conv7 = conv2d_bn(pool2, nb_filter * 16, kshape, stride=1, activ_fn='relu', border_mode='same',
         bias=False)

    for j in range(20):
        conv7 =inception_resnet(conv7, kshape, kshape2, nb_filter, subsample=False)

    print(conv7.shape)
    if reduction:
        conv8 = reduction_block(conv7,kshape,kshape2,nb_filter * 32)
    else:
        conv8 = conv2d_bn(conv7, nb_filter * 64, kshape, stride=1, activ_fn='relu', border_mode='same',
                      bias=False)
    for i in range(9):
        conv8 =inception_resnet(conv8, kshape, kshape2, nb_filter , subsample=False)
    conv9 = inception_resnet(conv8, kshape, kshape2, nb_filter, subsample=True)
    print(conv9.shape)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv8], axis=-1)
    print(up1.shape)
    for k in range(2):
        up1=conv2d_bn_drop(up1, nb_filter * 8, kshape, stride=1, activ_fn='relu', border_mode='same',
                     bias=False)
    up2 =UpSampling2D(size=(2, 2))(up1)
    # up2 =ZeroPadding2D(padding=((1,0),(1,0)))(up2)
    up2 = concatenate([up2, conv8], axis=-1)
    print(up2.shape)
    for k in range(2):
        up2=conv2d_bn_drop(up2, nb_filter * 8, kshape, stride=1, activ_fn='relu', border_mode='same',
                     bias=False)

    up3 = UpSampling2D(size=(2, 2))(up2)
    # up3 = ZeroPadding2D(padding=(1,1))(up3)
    print(up3.shape)
    up3 = concatenate([up3, conv5], axis=-1)
    for k in range(2):
        up3=conv2d_bn_drop(up3, nb_filter * 8, kshape, stride=1, activ_fn='relu', border_mode='same',
                    bias=False)
    up4 = UpSampling2D(size=(2, 2))(up3)
    # up4 = ZeroPadding2D(padding=((3, 2), (3, 2)))(up4)
    print(up4.shape)
    up4 = concatenate([up4, conv2], axis=-1)
    for k in range(2):
        up4 = conv2d_bn_drop(up4, nb_filter * 8, kshape, stride=1, activ_fn='relu', border_mode='same',
                          bias=False)
    # up5 = ZeroPadding2D(padding=(1, 1))(up4)
    up5 = concatenate([up4, conv1], axis=-1)
    for k in range(2):
        up5 = conv2d_bn_drop(up5, nb_filter * 4, kshape, stride=1, activ_fn='relu', border_mode='same',
                           bias=False)
    up6 = UpSampling2D(size=(2, 2))(up5)
    # up6 = ZeroPadding2D(padding=(1, 1))(up6)
    up6 = concatenate([up6, inputs], axis=-1)
    for k in range(2):
        up6 = conv2d_bn_drop(up6, nb_filter * 4, kshape, stride=1, activ_fn='relu', border_mode='same',
                            bias=False)
    final =conv2d_bn(up6, 1, kshape2, stride=1, activ_fn='relu', border_mode='same',
                     bias=False)
    final = Add()([final, inputs])

    model = Model(inputs=inputs, outputs=final)
    return model
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
# model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])
# model.summary()
def my_ssim_loss(y_true, y_pred):


    print(type(y_pred))
    print('y_shape:', y_true.shape)
    with tf.Session():
        y_pred = y_pred.eval()
        y_true = y_true.eval()

    #    y_true=np.array(y_true)
    #    y_pred=np.array(y_pred)
    print(type(y_pred))

    print('y_shape:', y_true.shape)
    shape = y_true.shape
    if len(shape) == 5:
        batch = shape[0]
        set_length = shape[1]

        total_loss = np.zeros((batch, set_length, 400, 400))

        for i in range(batch):
            for j in range(set_length):
                ssim_loss = 1 - compare_ssim(np.array(y_true[i, j]), np.array(y_pred[i, j]), multichannel=True)
                total_loss[i, j, :, :] += ssim_loss
        #                loss.append(ssim_loss)

        #        loss=np.array(loss).reshape(batch,set_lenth)

        total_loss = tf.convert_to_tensor(total_loss)
        print(type(total_loss))

        print('total_loss_shape:', total_loss.shape)
        return total_loss

def ssim_mse_loss(y_true, y_pred):
    # print(type(y_pred))
    # print('y_shape:', y_true.shape)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim_loss = 1 - ssim / denom
    mseloss = K.mean(K.square(y_pred - y_true), axis=-1)
    ssim_mse_loss = ssim_loss + mseloss
    return ssim_mse_loss

def msessimbce_loss(y_true, y_pred):
    # vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    # vgg = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    #     # loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    #     # loss_model.trainable = False
    #     # perceptual_loss = K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
    mseloss = K.mean(K.square(y_pred - y_true), axis=-1)
    ssim = 1 - tf.image.ssim(y_true, y_pred, 255.)
    # ssimloss = 1 - K.image.ssim(y_true, y_pred, max_val=255)
    # demon_true = K.sqrt(K.mean(K.square(y_true), axis=(1, 2, 3)))
    # demon_pred = K.sqrt(K.mean(K.square(y_pred), axis=(1, 2, 3)))
    # demon_true2 = K.sqrt(K.mean(K.square(y_true), axis=(0, 1, 2, 3)))
    # demon_pred2 = K.sqrt(K.mean(K.square(y_pred), axis=(0, 1, 2, 3)))
    # e_true = y_true / demon_true2 * K.log(K.clip((y_true / demon_true2), K.epsilon(), None))
    # e_pred = y_pred / demon_pred2 * K.log(K.clip((y_pred / demon_pred2), K.epsilon(), None))
    # bce = K.mean(e_true - e_pred, axis=(1, 2, 3))
    msessimbce_loss =  ssim*10
    return msessimbce_loss

def DnCNN(depth, filters=64, image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None, None, image_channels), name='input' + str(layer_count))
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

model = resnetv2_for_motion_normalsize()
model.summary()
# model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])


model.compile(loss ='mean_squared_error', optimizer='Nadam', metrics=['mse'])

itr = 1000
S = []
for i in range(itr):
    print("iteration = ", i + 1)
    if i < 500:
        bs = 4
    elif i < 2000:
        bs = 4
    elif i < 5000:
        bs = 4
    else:
        bs = 4
# img1 = loadmat('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_trainMotiondata_moco1.mat')
# img2 = loadmat('E:/chenyalei/deblur-gan-master/real_imag_moco_dataset/real_imag_trainGTdata_moco1.mat')
    train_X, train_Y = batch_data(n, batch_size=bs)
# train_X, train_Y = img1,img2
    history = model.fit(train_X, train_Y, validation_split=0.2 ,epochs=1, verbose=1)
#
# save_model(model,'unet423_1.h5')
    if i % 100 == 99:
        save_model(model, 'inception_resnet_0601.h5')

with open('trainHistoryDict10.txt', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

net_architecture = 'inception_resnet_0601'
# model = load_model('unet423_2.h5')
history_loss = history.history['loss']
history_val_loss = history.history['val_loss']
# axloss.plot(history_loss)
np_loss = np.array(history_loss)
np_val_loss = np.array(history_val_loss)
filePath = 'E:/chenyalei/u-net-master/u-net-master/loss'
filename_loss = filePath + net_architecture + '_loss.txt'
filename_val_loss = filePath + net_architecture + '_val_loss.txt'
np.savetxt(filename_loss,np_loss)
np.savetxt(filename_val_loss,np_val_loss)


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