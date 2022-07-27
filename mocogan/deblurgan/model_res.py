import tensorflow as tf
from keras.layers import Input, Activation, Add, UpSampling2D,concatenate
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.core import Dense, Flatten, Lambda


from keras.backend.common import normalize_data_format
from keras.models import Model
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers.merge import Add
from keras.utils import conv_utils

# from deblurgan.layer_utils import ReflectionPadding2D, res_block
from keras.layers.core import Dropout

# the paper defined hyper-parameter:chr
channel_rate = 64
# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 1)
patch_shape = (channel_rate, channel_rate, 1)

ngf = 32
ndf = 64
input_nc = 1
output_nc = 1
input_shape_generator = (256, 256, input_nc)
input_shape_discriminator = (256, 256, output_nc)
n_blocks_gen = 7

def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False):
    """
    Instanciate a Keras Resnet Block using sequential API.

    :param input: Input tensor
    :param filters: Number of filters to use
    :param kernel_size: Shape of the kernel for the convolution
    :param strides: Shape of the strides for the convolution
    :param use_dropout: Boolean value to determine the use of dropout
    :return: Keras Model
    """
    x = ReflectionPadding2D((1, 1))(input)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,)(x)
    x = BatchNormalization()(x)

    merged = Add()([input, x])
    return merged

def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """
    Pad the 2nd and 3rd dimensions of a 4D tensor.

    :param x: Input tensor
    :param padding: Shape of padding to use
    :param data_format: Tensorflow vs Theano convention ('channels_last', 'channels_first')
    :return: Tensorflow tensor
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if data_format == 'channels_first':
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]), list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")



class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns or zeros
    at the top, bottom, left and right side of an image tensor.
    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to width and height.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """

    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                             padding=self.padding,
                                             data_format=self.data_format)

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def generator_model():
    """Build generator architecture."""
    # Current version : ResNet block
    inputs = Input(shape=image_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=32, kernel_size=(7, 7), padding='valid',strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    # n_downsampling = 5
    # for i in range(n_downsampling):
    # mult = 2**i
    en_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same')(x)
    en_1 = BatchNormalization()(en_1)
    en_1 = Activation('relu')(en_1)

    en_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same')(en_1)
    en_2 = BatchNormalization()(en_2)
    en_2 = Activation('relu')(en_2)

    en_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same')(en_2)
    en_3 = BatchNormalization()(en_3)
    en_3 = Activation('relu')(en_3)

    en_4 = Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same')(en_3)
    en_4 = BatchNormalization()(en_4)
    en_4 = Activation('relu')(en_4)

    en_5 = Conv2D(filters=1024, kernel_size=(3, 3), strides=2, padding='same')(en_4)
    en_5 = BatchNormalization()(en_5)
    en_5 = Activation('relu')(en_5)

    # en_6 = Conv2D(filters=1024, kernel_size=(3, 3), strides=2, padding='same')(en_5)
    # en_6 = BatchNormalization()(en_6)
    # en_6 = Activation('relu')(en_6)



    # mult = 2**n_downsampling
    for i in range(n_blocks_gen):
        # x = res_block(x, ngf*mult, use_dropout=True)
        en_5 = res_block(en_5, 1024, use_dropout=True)
    #
    # for i in range(n_downsampling):
    #     mult = 2**(n_downsampling - i)
    #     # x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
    #     x = UpSampling2D()(x)
    #     x = Conv2D(filters=int(ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    up_1 = UpSampling2D()(en_5)
    up_1 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(up_1)
    merge_1 = concatenate([en_4, up_1], axis=3)
    de_1 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(merge_1)
    de_1 = BatchNormalization()(de_1)
    de_1 = Activation('relu')(de_1)

    up_2 = UpSampling2D()(de_1)
    up_2 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(up_2)
    merge_2 = concatenate([en_3, up_2], axis=3)
    de_2 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(merge_2)
    de_2 = BatchNormalization()(de_2)
    de_2 = Activation('relu')(de_2)

    up_3 = UpSampling2D()(de_2)
    up_3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(up_3)
    merge_3 = concatenate([en_2, up_3], axis=3)
    de_3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(merge_3)
    de_3 = BatchNormalization()(de_3)
    de_3 = Activation('relu')(de_3)

    up_4 = UpSampling2D()(de_3)
    up_4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(up_4)
    merge_4 = concatenate([en_1, up_4], axis=3)
    de_4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(merge_4)
    de_4 = BatchNormalization()(de_4)
    de_4 = Activation('relu')(de_4)

    up_5 = UpSampling2D()(de_4)
    up_5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(up_5)
    merge_5 = concatenate([x, up_5], axis=3)
    de_5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(merge_5)
    de_5 = BatchNormalization()(de_5)
    de_5 = Activation('relu')(de_5)

    # up_6 = UpSampling2D()(de_5)
    # up_6 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(up_6)
    # merge_6 = concatenate([x, up_6], axis=3)
    # de_6 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(merge_6)
    # de_6 = BatchNormalization()(de_6)
    # de_6 = Activation('relu')(de_6)
    # up_7 = UpSampling2D()(de_6)
    # up_7 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(up_7)
    output = ReflectionPadding2D((3, 3))(de_5)
    output = Conv2D(filters=output_nc, kernel_size=(7, 7), padding='valid')(output)
    output = Activation('tanh')(output)
    del en_1, en_2, en_3, en_4, en_5
    del up_1, up_2, up_3, up_4, up_5, merge_1, merge_2, merge_3, merge_4, merge_5

    outputs = Add()([output, inputs])
    # outputs = Lambda(lambda z: K.clip(z, -1, 1))(x)
    outputs = Lambda(lambda z: z/2)(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model


def discriminator_model():
    """Build discriminator architecture."""
    n_layers, use_sigmoid = 2, False
    inputs = Input(shape=input_shape_discriminator)

    x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)


    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

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
