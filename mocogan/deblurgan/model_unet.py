from keras.layers import Input, Activation, Add, UpSampling2D,Conv2DTranspose,MaxPooling2D,concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
# from keras.backend.common import normalize_data_format
# from .layer_utils import ReflectionPadding2D, res_block
from keras.layers.core import Dropout

# the paper defined hyper-parameter:chr
channel_rate = 64
# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)
patch_shape = (channel_rate, channel_rate, 3)

ngf = 64
ndf = 64
input_nc = 3
output_nc = 3
input_shape_generator = (256, 256, input_nc)
input_shape_discriminator = (256, 256, output_nc)
n_blocks_gen = 6

def encoder_block(x,k):
    x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x
def decoder_block(x,k):
    # x = PixelShuffler()(x)
    x = Conv2D(filters=k, kernel_size=1, strides=1, padding='same')(x)

    x = UpSampling2D(size=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def D_conv(x,k,n,use_normalization):
    x = Conv2D(filters=k,kernel_size=n,strides=1,padding='same')(x)
    if use_normalization:
        x = BatchNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def u_block(inputs):
    conv1 = encoder_block(inputs, 32)
    # Layer 2
    conv2 = encoder_block(conv1, 64)
    # Layer 3
    conv3 = encoder_block(conv2, 128)
    # Layer 4
    conv4 = encoder_block(conv3, 256)
    conv5 = encoder_block(conv4, 256)
    conv6 = encoder_block(conv5, 256)
    conv7 = encoder_block(conv6, 256)
    conv8 = encoder_block(conv7, 256)
    up1 = decoder_block(conv8, 256)
    merge1 = concatenate([conv7, up1], axis=3)
    up2 = decoder_block(merge1, 512)
    merge2 = concatenate([conv6, up2], axis=3)
    up3 = decoder_block(merge2, 512)
    merge3 = concatenate([conv5, up3], axis=3)
    up4 = decoder_block(merge3, 512)
    merge4 = concatenate([conv4, up4], axis=3)
    up5 = decoder_block(merge4, 512)
    merge5 = concatenate([conv3, up5], axis=3)
    up6 = decoder_block(merge5, 256)
    merge6 = concatenate([conv2, up6], axis=3)
    up7 = decoder_block(merge6, 128)
    merge7 = concatenate([conv1, up7], axis=3)
    up8 = decoder_block(merge7, 64)
    # x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(output_nc, 1, strides=1)(up8)
    x = Activation('tanh')(x)  # They say they use Relu but really they do not
    del conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8
    del up1, up2, up3, up4, up5, up6, up7, up8, merge1, merge2, merge3, merge4, merge5, merge6
    outputs = x
    return outputs
def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def generator_model():



    inputs = Input(shape=image_shape)

    u_block_outputs = u_block(inputs)
    u_block_outputs = u_block(u_block_outputs)
    u_block_outputs = u_block(u_block_outputs)
    u_block_outputs = u_block(u_block_outputs)
    u_block_outputs = u_block(u_block_outputs)
    u_block_outputs = u_block(u_block_outputs)

    model = Model(inputs=inputs, outputs=u_block_outputs, name='Generator')
    return model


def discriminator_model():
    """Build discriminator architecture."""
    # n_layers, use_sigmoid = 1, False
    inputs = Input(shape=input_shape_discriminator)

    x = Conv2D(filters=64, kernel_size=(16, 16), strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=128, kernel_size=(16, 16), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=64, kernel_size=(8, 8), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=128, kernel_size=(8, 8), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)




    # nf_mult, nf_mult_prev = 1, 1
    # for n in range(n_layers):
    #     nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
    #     x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = LeakyReLU(0.2)(x)

    # nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    # x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)

    # x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)
    # if use_sigmoid:
    #     x = Activation('sigmoid')(x)
    x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model


def generator_containing_discriminator(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return model


if __name__ == '__main__':
    g = generator_model()
    g.summary()
    d = discriminator_model()
    d.summary()
    m = generator_containing_discriminator(generator_model(), discriminator_model())
    m.summary()
