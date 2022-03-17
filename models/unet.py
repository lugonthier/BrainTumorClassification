import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D,
                                     BatchNormalization,ReLU)
from blocks.up_and_down_sampling import DownSamplingBlock, UpSamplingBlock


class U_Net(tf.keras.Model):

    def __init__(self, input_shape, n_classes, name='U-Net', **kwargs):
        super(U_Net, self).__init__(**kwargs)

        self.in_shape = input_shape

        # Downsampling Module

        # Entry block
        self.entry_conv = Conv2D(32, 3, strides=2, padding='same')
        self.entry_batch = BatchNormalization()
        self.entry_activ = ReLU()

        # Down sampling Module
        self.down_block1 = DownSamplingBlock(n_features=64)
        self.down_block2 = DownSamplingBlock(n_features=128)
        self.down_block3 = DownSamplingBlock(n_features=256)

        # Up sampling Module
        self.up_block1 = UpSamplingBlock(n_features=256)
        self.up_block2 = UpSamplingBlock(n_features=128)
        self.up_block3 = UpSamplingBlock(n_features=64)
        self.up_block4 = UpSamplingBlock(n_features=32)

        self.output_conv = Conv2D(n_classes, 3, activation='softmax', padding='same')

    def call(self, inputs, training=None, mask=None):
        x = self.entry_conv(inputs)
        x = self.entry_batch(x)
        x = self.entry_activ(x)


        x = self.down_block1(x)
        x = self.down_block2(x)
        x = self.down_block3(x)

        x = self.up_block1(x)
        x = self.up_block2(x)
        x = self.up_block3(x)
        x = self.up_block4(x)

        return self.output_conv(x)

    def build_graph(self):
        x = Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == '__main__':
    import numpy as np
    X = np.random.rand(10, 160, 160, 3)
    print(X.shape)
    y = np.random.randint(0, 2, (10, 160, 160, 1))
    y = tf.keras.utils.to_categorical(y, 2)
    print(y)
    model = U_Net((160, 160, 3), 2)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)


    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("test_segmentation.h5", save_best_only=True)
    ]

    epochs = 5
    model.fit(X, y, epochs=epochs, batch_size=2, callbacks=callbacks)