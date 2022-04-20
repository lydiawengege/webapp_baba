from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import add
from keras.engine import InputSpec
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, UpSampling3D
from keras.layers import BatchNormalization
from keras import regularizers

import tensorflow as tf

reg_weights = 0.001

def bn_relu():
    def bn_relu_func(x):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return bn_relu_func


def conv_bn_relu(nb_filter, nb_row, nb_col, nb_dep, stride):
    def conv_func(x):
        x = Conv3D(nb_filter, (nb_row, nb_col, nb_dep), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return conv_func


def res_conv(nb_filter, nb_row, nb_col, nb_dep, stride=(1, 1, 1)):
    def _res_func(x):
        identity = x

        a = Conv3D(nb_filter, (nb_row, nb_col, nb_dep), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(x)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Conv3D(nb_filter, (nb_row, nb_col, nb_dep), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(a)
        y = BatchNormalization()(a)

        return add([identity, y])

    return _res_func


def dconv_bn_nolinear(nb_filter, nb_row, nb_col, nb_dep, stride=(2, 2, 2), activation="relu"):
    def _dconv_bn(x):
        x = UpSampling3D(size=stride)(x)
        x = ReflectionPadding3D(padding=(int(nb_row/2), int(nb_col/2), int(nb_dep/2)))(x)
        x = Conv3D(nb_filter, (nb_row, nb_col, nb_dep), padding='valid', kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    return _dconv_bn


class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), dim_ordering='default', **kwargs):
        super(ReflectionPadding3D, self).__init__(**kwargs)

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.padding = padding
        if isinstance(padding, dict):
            if set(padding.keys()) <= {'top_pad', 'bottom_pad', 'left_pad', 'right_pad', 'front_pad', 'back_pad'}:
                self.top_pad = padding.get('top_pad', 0)
                self.bottom_pad = padding.get('bottom_pad', 0)
                self.left_pad = padding.get('left_pad', 0)
                self.right_pad = padding.get('right_pad', 0)
                self.front_pad = padding.get('front_pad', 0)
                self.back_pad = padding.get('back_pad', 0)
            else:
                raise ValueError('Unexpected key found in `padding` dictionary. '
                                 'Keys have to be in {"top_pad", "bottom_pad", '
                                 '"left_pad", "right_pad", "front_pad", "back_pad"}.'
                                 'Found: ' + str(padding.keys()))
        else:
            padding = tuple(padding)
            if len(padding) == 3:
                self.top_pad = padding[0]
                self.bottom_pad = padding[0]
                self.left_pad = padding[1]
                self.right_pad = padding[1]
                self.front_pad = padding[2]
                self.back_pad = padding[2]
            elif len(padding) == 6:
                self.top_pad = padding[0]
                self.bottom_pad = padding[1]
                self.left_pad = padding[2]
                self.right_pad = padding[3]
                self.front_pad = padding[4]
                self.back_pad = padding[5]
            else:
                raise TypeError('`padding` should be tuple of int '
                                'of length 3 or 6, or dict. '
                                'Found: ' + str(padding))

        if dim_ordering not in {'tf'}:
            raise ValueError('dim_ordering must be in {tf}.')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=5)]

    def call(self, x, mask=None):
        top_pad = self.top_pad
        bottom_pad = self.bottom_pad
        left_pad = self.left_pad
        right_pad = self.right_pad
        front_pad = self.front_pad        
        back_pad = self.back_pad
        
        paddings = [[0, 0], [left_pad, right_pad], [top_pad, bottom_pad], [front_pad, back_pad], [0, 0]]

        return tf.pad(x, paddings, mode='REFLECT', name=None)

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'tf':
            rows = input_shape[1] + self.top_pad + self.bottom_pad if input_shape[1] is not None else None
            cols = input_shape[2] + self.left_pad + self.right_pad if input_shape[2] is not None else None
            dep = input_shape[3] + self.front_pad + self.back_pad if input_shape[3] is not None else None

            return (input_shape[0],
                    rows,
                    cols,
                    dep,
                    input_shape[4])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class UnPooling2D(UpSampling2D):
#     def __init__(self, size=(2, 2)):
#         super(UnPooling2D, self).__init__(size)

#     def call(self, x, mask=None):
#         shapes = x.get_shape().as_list()
#         w = self.size[0] * shapes[1]
#         h = self.size[1] * shapes[2]
#         return tf.image.resize_nearest_neighbor(x, (w, h))


# class InstanceNormalize(Layer):
#     def __init__(self, **kwargs):
#         super(InstanceNormalize, self).__init__(**kwargs)
#         self.epsilon = 1e-3

#     def call(self, x, mask=None):
#         mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
#         return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, self.epsilon)))

#     def compute_output_shape(self, input_shape):
#         return input_shape