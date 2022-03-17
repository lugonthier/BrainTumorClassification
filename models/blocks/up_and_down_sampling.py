import tensorflow as tf

from tensorflow.keras.layers import (Layer, Conv2D, BatchNormalization,
                                     ReLU, SeparableConv2D,
                                     MaxPooling2D,
                                     Conv2DTranspose,
                                     UpSampling2D)

class DownSamplingBlock(Layer):

    def __init__(self, n_features , name='Unet_DownSamplingBlock', **kwargs):
        super(DownSamplingBlock, self).__init__(**kwargs)
        self.activ1 = ReLU()
        self.sep_conv1 = SeparableConv2D(n_features, 3, padding='same')
        self.batch1 = BatchNormalization()

        self.activ2 = ReLU()
        self.sep_conv2 = SeparableConv2D(n_features, 3, padding='same')
        self.batch2 = BatchNormalization()
        self.pooling = MaxPooling2D(3, strides=2, padding='same')

        self.residual_conv = Conv2D(n_features, 1, strides=2, padding='same')


    def call(self, inputs, training=True):

        x = self.activ1(inputs)
        x = self.sep_conv1(x)
        x = self.batch1(x)

        x = self.activ2(x)
        x = self.sep_conv2(x)
        x = self.batch2(x)
        x = self.pooling(x)

        # Residual
        residual = self.residual_conv(inputs)
        x = tf.keras.layers.add([x, residual])

        return x



class UpSamplingBlock(Layer):
    def __init__(self, n_features, name='Unet_UpSampling_block', **kwargs):
        super(UpSamplingBlock, self).__init__(**kwargs)

        self.activ1 = ReLU()
        self.conv_trans1 = Conv2DTranspose(n_features, 3, padding='same')
        self.batch1 = BatchNormalization()

        self.activ2 = ReLU()
        self.conv_trans2 = Conv2DTranspose(n_features, 3, padding='same')
        self.batch2 = BatchNormalization()

        self.up_sampling1 = UpSampling2D(2)

        self.residual_up_sampling = UpSampling2D(2)
        self.residual_conv = Conv2D(n_features, 1, padding='same')


    def call(self, inputs, training=True):

        x = self.activ1(inputs)
        x = self.conv_trans1(x)
        x = self.batch1(x)

        x = self.activ2(x)
        x = self.conv_trans2(x)
        x = self.batch2(x)

        x = self.up_sampling1(x)

        # Residual
        residual = self.up_sampling1(inputs)
        residual = self.residual_conv(residual)
        x = tf.keras.layers.add([x, residual])

        return x