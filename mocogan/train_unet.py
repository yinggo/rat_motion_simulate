import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import datetime
import click
import numpy as np
import tqdm
import tensorflow as tf

from deblurgan.utils import load_images, write_log
from deblurgan.losses import wasserstein_loss, perceptual_loss,l1_loss
from deblurgan.model_unet import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs
# from keras.backend.common import normalize_data_format
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

BASE_DIR = 'weights/'

def gasuss_noise(image, mean=0, var=0.5):
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

def num_patches(output_img_dim=(256,256,3),sub_patch_dim=(64,64)):
    nb_non_overlaping_patching = (output_img_dim[0]/sub_patch_dim[0]*(output_img_dim[1]/sub_patch_dim[1]))
    patch_disc_img_dim = (sub_patch_dim[0],sub_patch_dim[1],output_img_dim[2])
    return int(nb_non_overlaping_patching),patch_disc_img_dim

def extract_patches(images, sub_patch_dim):
    """
    Cuts images into k subpatches
    Each kth cut as the kth patches for all images
    ex: input 3 images [im1, im2, im3]
    output [[im_1_patch_1, im_2_patch_1], ... , [im_n-1_patch_k, im_n_patch_k]]

    :param images: array of Images (num_images, im_channels, im_height, im_width)
    :param sub_patch_dim: (height, width) ex: (30, 30) Subpatch dimensions
    :return:
    """

    im_height, im_width = images.shape[1:3]
    patch_height, patch_width = sub_patch_dim

    # list out all xs  ex: 0, 29, 58, ...
    x_spots = range(0, im_width, patch_width)

    # list out all ys ex: 0, 29, 58
    y_spots = range(0, im_height, patch_height)
    all_patches = []

    for y in y_spots:
        for x in x_spots:
            # indexing here is cra
            # images[num_images, num_channels, width, height]
            # this says, cut a patch across all images at the same time with this width, height
            image_patches = images[:, x: x+patch_width, y: y+patch_height, :]
            all_patches.append(np.asarray(image_patches, dtype=np.float32))
    return all_patches


def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


def train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates=5):
    x_train_1 = np.load('/home/user/data1/yalei/hcp_motion_1-3.npy')
    y_train_1 = np.load('/home/user/data1/yalei/hcp_GT_1-3.npy')
    x_train_2 = np.load('/home/user/data1/yalei/hcp_motion_4-6.npy')
    y_train_2 = np.load('/home/user/data1/yalei/hcp_GT_4-6.npy')
    x_train_3 = np.load('/home/user/data1/yalei/hcp_motion_7-9.npy')
    y_train_3 = np.load('/home/user/data1/yalei/hcp_GT_7-9.npy')
    # x_train_4 = np.load('/home/user/data1/yalei/hcp_motion_10-11.npy')
    # y_train_4 = np.load('/home/user/data1/yalei/hcp_GT_10-11.npy')
    # x_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/knee_data/knee_motion/knee_motion_629.npy')
    # y_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/knee_data/knee_gt/knee_gt_629.npy')
    x_train_4 = np.load('/home/user/data1/yalei/hcp_motion_13_17.npy')
    y_train_4 = np.load('/home/user/data1/yalei/hcp_GT_13_17.npy')

    # a = x_train_3[0,:,:]
    # b = x_train_4[0,:,:]
    # c = y_train_4[0,:,:]
    # x_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/motion_data_611/noisy_motion.npy')
    # y_train_4 = np.load('E:/chenyalei/deblur-gan-master/dataset/GT_data_611/noise_gt.npy')
    x_train = np.concatenate((x_train_1,x_train_2,x_train_3,x_train_4),axis=0)
    y_train = np.concatenate((y_train_1,y_train_2,y_train_3,y_train_4),axis=0)

    for i in range(x_train.shape[0]):
        x_train[i,:,:] = (x_train[i,:,:] - 127.5)/127.5
        y_train[i,:,:] = (y_train[i,:,:] - 127.5)/127.5

    # for i in range(x_train_4.shape[0]):
    #     x_train_4[i,:,:] = x_train_4[i,:,:]*10
    #     y_train_4[i,:,:] = y_train_4[i,:,:]*10
    # # a = x_train_4[0,:,:]
    # x_train = np.concatenate((x_train_4,x_train),axis=0)
    # y_train = np.concatenate((y_train_4,y_train),axis=0)
    # a = x_train[0,:,:]
    # x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
    # y_train = np.reshape(y_train,(y_train.shape[0],y_train.shape[1],y_train.shape[2],1))
    y_train = y_train.astype('float64')
    x_train_channel3 = np.zeros([x_train.shape[0], 256, 256, 3],'float64')
    y_train_channel3 = np.zeros([y_train.shape[0], 256, 256, 3],'float64')
    for channel in range(y_train_channel3.shape[3]):
        x_train_channel3[:, :, :, channel] = x_train
        y_train_channel3[:, :, :, channel] = y_train
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
    d = discriminator_model()
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

    d_opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    output_true_batch, output_false_batch = 0.9*np.ones((batch_size, 1)), 0.9*(-np.ones((batch_size, 1)))

    log_path = './logs'
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
            image_blur_batch = x_train_channel3[batch_indexes]
            image_full_batch = y_train_channel3[batch_indexes]

            # test_image_blur_batch = x_test[batch_indexes]
            # test_image_full_batch = y_test[batch_indexes]
            # image_full_batch = gasuss_noise(image_full_batch)
            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            # generated_images = gasuss_noise(generated_images)
            for _ in range(critic_updates):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)
                d_losses_fake.append(d_loss_fake)
                d_losses_real.append(d_loss_real)

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            d_on_g_losses.append(d_on_g_loss)
            # test_generated_images = g.predict(x=test_image_blur_batch,batch_size=batch_size)
            # test_loss = d_on_g.test_on_batch(test_generated_images,[test_image_full_batch,output_true_batch])
            # test_losses.append(test_loss)
            d.trainable = True

        # write_log(tensorboard_callback, ['g_loss', 'd_on_g_loss'], [np.mean(d_losses), np.mean(d_on_g_losses)], epoch_num)
        print(np.mean(d_losses), np.mean(d_losses_fake),np.mean(d_losses_real),np.mean(d_on_g_losses))
        with open('log_715_medgan_epoch100.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))


@click.command()
@click.option('--n_images', default=-1, help='Number of images to load for training')
@click.option('--batch_size', default=8, help='Size of batch')
@click.option('--log_dir', required=True, default='E:/chenyalei/deblur-gan-master/logs',help='Path to the log_dir for Tensorboard')
@click.option('--epoch_num', default=100, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, log_dir, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates)


if __name__ == '__main__':
    K.tensorflow_backend._get_available_gpus()
    train_command()
