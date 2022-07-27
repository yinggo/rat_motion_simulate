import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import datetime
import click
import numpy as np
import tqdm
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

from keras.utils import multi_gpu_model
from deblurgan.utils import load_images, write_log
from deblurgan.losses import wasserstein_loss, perceptual_loss,l1_loss,perceptual_l1_loss,edge_l1_loss,entropy_l1_loss,entropy_loss2
from deblurgan.model_res import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs
from keras.backend.common import normalize_data_format
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from keras.preprocessing.image import ImageDataGenerator
from numpy import moveaxis

BASE_DIR = 'weights_mild_mouse/'

def gasuss_noise(image, mean=0, var=0.0005):
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

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    # d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


def train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates=5):

    # x_train_1 = np.load('/home/user/data1/yalei/hcp_motion_1-3.npy')
    # y_train_1 = np.load('/home/user/data1/yalei/hcp_GT_1-3.npy')
    # x_train_2 = np.load('/home/user/data1/yalei/hcp_motion_4-6.npy')
    # y_train_2 = np.load('/home/user/data1/yalei/hcp_GT_4-6.npy')
    # x_train_3 = np.load('/home/user/data1/yalei/hcp_motion_7-9.npy')
    # y_train_3 = np.load('/home/user/data1/yalei/hcp_GT_7-9.npy')
    # # x_train_4 = np.load('/home/user/data1/yalei/hcp_motion_10-11.npy')
    # # y_train_4 = np.load('/home/user/data1/yalei/hcp_GT_10-11.npy')
    # # x_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/knee_data/knee_motion/knee_motion_629.npy')
    # # y_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/knee_data/knee_gt/knee_gt_629.npy')
    # x_train_4 = np.load('/home/user/data1/yalei/hcp_motion_13_17.npy')
    # y_train_4 = np.load('/home/user/data1/yalei/hcp_GT_13_17.npy')
    # x_train_1 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_1.npy')
    # # x_train_1 = cv2.normalize(x_train_1, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
    # # plt.imshow(x_train_1[10,:,:],cmap='gray')
    # # plt.show()
    #
    #
    #
    # # a = x_train_1[8,:,:]
    # y_train_1 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_1.npy')
    #
    # x_train_2 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_2.npy')
    # y_train_2 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_2.npy')
    #
    # x_train_3 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_3.npy')
    # y_train_3 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_3.npy')
    #
    # x_train_4 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_4.npy')
    # y_train_4 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_4.npy')
    #
    # x_train_5 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_5.npy')
    # y_train_5 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_5.npy')
    #
    # x_train_6 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_6.npy')
    # y_train_6 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_6.npy')
    #
    # x_train_7 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_7.npy')
    # y_train_7 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_7.npy')
    #
    # x_train_8 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_8.npy')
    # y_train_8 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_8.npy')
    #
    # x_train_9 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_9.npy')
    # y_train_9 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_9.npy')
    #
    # x_train_10 = np.load('/home/user/data1/mouse_data/DataSetRat/gt/rat_motion_10.npy')
    # y_train_10 = np.load('/home/user/data1/mouse_data/DataSetRat/motion/rat_GT_10.npy')
    # # a = x_train_3[0,:,:]
    # # b = x_train_4[0,:,:]
    # # c = y_train_4[0,:,:]
    # # x_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/noisy_motion.npy')
    # # y_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/GT_data_611/noise_gt.npy')
    # x_train = np.concatenate((x_train_1,x_train_2,x_train_3,x_train_4,x_train_5,x_train_6,x_train_7,x_train_8,x_train_9,x_train_10),axis=0)
    # y_train = np.concatenate((y_train_1,y_train_2,y_train_3,y_train_4,y_train_5,y_train_6,y_train_7,y_train_8,y_train_9,y_train_10),axis=0)
    # rootdir1 = '/home/user/data1/mouse_data/FullBrain/train/gt/'
    # rootdir2 = '/home/user/data1/mouse_data/FullBrain/train/motion/'
    # list1 = os.listdir(rootdir1)
    # list2 = os.listdir(rootdir2)
    # list1.sort(key=lambda x: int(x[24:-4]))
    # list2.sort(key=lambda x: int(x[20:-4]))
    #
    # y_train_1 = np.zeros(shape=(1, 256, 256), dtype='float64')
    # x_train_1 = np.zeros(shape=(1, 256, 256), dtype='float64')
    # # x_train_1 = np.array(x_train_1)
    # # y_train_1 = np.array(y_train_1)
    # # y_train_2 = np.zeros(shape=(19000, 256, 256), dtype='float64')
    # # x_train_2 = np.zeros(shape=(19000, 256, 256), dtype='float64')
    # # for num in range(0, len(list1)):
    # #     path1 = os.path.join(rootdir1, list1[num])
    # #     motion_img = sio.loadmat(path1)
    # #     motion_img = abs(motion_img['data'])
    # #
    # #     x_train_1[num * 300:(num + 1) * 300, :, :] = motion_img
    # #     path2 = os.path.join(rootdir2, list2[num])
    # #     GT1 = sio.loadmat(path2)
    # #     GT1 = GT1['data']
    # #     y_train_1[num * 300:(num + 1) * 300, :, :] = GT1
    # # for num in range(0, len(list1)):
    # for num in range(0, 1):
    #     path1 = os.path.join(rootdir1, list1[num])
    #     motion_img = sio.loadmat(path1)
    #     motion_img = abs(motion_img['data1'])
    #
    #     x_train_1 = np.concatenate((x_train_1,motion_img),axis=0)
    #
    #     path2 = os.path.join(rootdir2, list2[num])
    #     GT1 = sio.loadmat(path2)
    #     GT1 = GT1['data2']
    #     y_train_1 = np.concatenate((y_train_1, GT1), axis=0)
    #     # y_train_1.append(GT1)
    # x_train = x_train_1[1:, :, :]
    # y_train = y_train_1[1:, :, :]
        # y_train_1[(num) * 100:(num+ 1) * 100, :, :] = GT1
    # x_train = np.concatenate((x_train_1, x_train_2), axis=0)
    # y_train = np.concatenate((y_train_1, y_train_2), axis=0)
    # x_mean = np.mean(x_train,axis=0)
    # y_mean = np.mean(y_train,axis=0)
    # x_std = np.std(x_train,axis=0)
    # y_std = np.std(y_train,axis=0)
    #
    # x_train = (x_train-x_mean)/x_std
    # y_train = (y_train-y_mean)/y_std
    # a = x_train[1,:,:]
    # b = y_train[1,:,:]

    # mouse data:
    # x_train_1 = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/motionChannel1.npy')
    # x_train_1 = x_train_1[0:2000,:,:]
    # y_train_1 = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/motionfreeChannel1.npy')
    # y_train_1 = y_train_1[0:2000,:,:]

    # hcp data:
    # x_train_1 = np.load('/home/user/data1/yalei/hcp_motion_1-3.npy')
    # y_train_1 = np.load('/home/user/data1/yalei/hcp_GT_1-3.npy')

    # mild mouse data:
    x_train_1 = sio.loadmat('/home/user/data2/bcx/mocogan_v2/scripts/mild_data/motion.mat')
    y_train_1 = sio.loadmat('/home/user/data2/bcx/mocogan_v2/scripts/mild_data/gt.mat')
    x_train = x_train_1['motion']
    y_train = y_train_1['gt']
    y_train = y_train[0:900, :, :]
    x_train = x_train.transpose(2, 1, 0)
    x_train = x_train[0:900, :, :]

    # motion = x_train_1
    # gt = y_train_1
    # sio.savemat('./motion', {'motion': motion})
    # sio.savemat('./gt',{'gt':gt})

    # x_train_2 = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/train_motion_markov_no_blur.npy')
    # y_train_2 = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/train_gt_markov_no_blur.npy')
    # x_train = np.concatenate((x_train_1, x_train_2), axis=0)
    # y_train = np.concatenate((y_train_1, y_train_2), axis=0)
    # x_train = x_train_1
    # y_train = y_train_1
    for i in range(x_train.shape[0]):
        x_train[i, :, :] = cv2.normalize(x_train[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
        y_train[i, :, :] = cv2.normalize(y_train[i, :, :], None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)


    # sio.savemat('./motion',{'motion':x_train})
    # sio.savemat('./gt',{'gt':y_train})

    # a = x_train[100,:,:]
    # b = y_train[100,:,:]
    # plt.imshow(np.concatenate((a,b),axis=0),cmap='gray')
    # plt.show()
    # # x_train = cv2.normalize(x_train, None, 0, 1, cv2.NORM_HAMMING)
    # for i in range(x_train.shape[0]):
    #     x_train[i,:,:] = (x_train[i,:,:] - 127.5)/127.5
    #     y_train[i,:,:] = (y_train[i,:,:] - 127.5)/127.5

    # for i in range(x_train_4.shape[0]):
    #     x_train_4[i,:,:] = x_train_4[i,:,:]*10
    #     y_train_4[i,:,:] = y_train_4[i,:,:]*10
    # # a = x_train_4[0,:,:]
    # x_train = np.concatenate((x_train_4,x_train),axis=0)
    # y_train = np.concatenate((y_train_4,y_train),axis=0)
    # a = x_train[0,:,:]
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
    y_train = np.reshape(y_train,(y_train.shape[0],y_train.shape[1],y_train.shape[2],1))
    # y_train = y_train.astype('float64')
    # x_test = np.load('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/hcp_motion_7-9.npy')
    # y_test = np.load('E:/chenyalei/deblur-gan-master/dataset/GT_data_611/hcp_GT_7-9.npy')
    # for i in range(x_test.shape[0]):
    #     x_test[i,:,:] = (x_test[i,:,:] - 127.5)/127.5
    #     y_test[i,:,:] = (y_test[i,:,:] - 127.5)/127.5
    # x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
    # y_test = np.reshape(y_test,(y_test.shape[0],y_test.shape[1],y_test.shape[2],1))
    # y_test = y_test.astype('float64')
    # data = load_images('./images/train', n_images)
    # y_train, x_train = data['B'], data['A']

    g = generator_model()
    # g_parallel = multi_gpu_model(g,gpus=2)
    d = discriminator_model()
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
    # d_on_g = g

    d_opt = Adam(lr=1E-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.999)
    d_on_g_opt = Adam(lr=1E-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.999)

    d.trainable = True
    # d_parallel=multi_gpu_model(d,gpus=2)
    # d_parallel.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_l1_loss, wasserstein_loss]
    loss_weights = [1000, 1]

    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    # d_on_g.compile(optimizer=d_on_g_opt, loss=entropy_loss2)
    d.trainable = True

    output_true_batch, output_false_batch = 0.9*np.ones((batch_size, 1)), 0.9*(-np.ones((batch_size, 1)))

    log_path = '/home/user/data2/bcx/mocogan_v2/scripts/logs'
    tensorboard_callback = TensorBoard(log_path)

    for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_losses_fake = []
        d_losses_real = []
        d_on_g_losses = []
        test_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]
            # test_image_blur_batch = x_test[batch_indexes]
            # test_image_full_batch = y_test[batch_indexes]

            # generated_images = g_parallel.predict(x=image_blur_batch, batch_size=batch_size)
            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)
            # image_full_batch = gasuss_noise(image_full_batch)
            # image_full_batch_noise = gasuss_noise(image_full_batch)
            # generated_images_noise = gasuss_noise(generated_images)
            # plt.imshow(np.concatenate((image_full_batch[0,:,:,0],generated_images[0,:,:,0]),axis=1),cmap='gray')
            # plt.show()
            for _ in range(critic_updates):
                # d_loss_real = d_parallel.train_on_batch(image_full_batch_noise, output_true_batch)
                # d_loss_fake = d_parallel.train_on_batch(generated_images_noise, output_false_batch)
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)
                d_losses_fake.append(d_loss_fake)
                d_losses_real.append(d_loss_real)

            # d_parallel.trainable = False
            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            # d_on_g_loss = d_on_g_parallel.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            d_on_g_losses.append(d_on_g_loss)
            # print(d_on_g_loss)
            # test_generated_images = g.predict(x=test_image_blur_batch,batch_size=batch_size)
            # test_loss = d_on_g.test_on_batch(test_generated_images,[test_image_full_batch,output_true_batch])
            # test_losses.append(test_loss)
            # d_parallel.trainable = True
            d.trainable = True

        # write_log(tensorboard_callback, ['g_loss', 'd_on_g_loss'], [np.mean(d_losses), np.mean(d_on_g_losses)], epoch_num)
        print(np.mean(d_losses), np.mean(d_losses_fake),np.mean(d_losses_real),np.mean(d_on_g_losses))
        with open('/home/user/data2/bcx/mocogan_v2/scripts/logs/log_mild_mouse_data.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))


@click.command()
@click.option('--n_images', default=-1, help='Number of images to load for training')
@click.option('--batch_size', default=2, help='Size of batch')
@click.option('--log_dir', required=True, default='E:/chenyalei/deblur-gan-master/logs',help='Path to the log_dir for Tensorboard')
@click.option('--epoch_num', default=100, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, log_dir, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates)


if __name__ == '__main__':
    train_command()
