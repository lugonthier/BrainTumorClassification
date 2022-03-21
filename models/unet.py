import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D,
                                     BatchNormalization,ReLU)
from models.blocks.up_and_down_sampling import DownSamplingBlock, UpSamplingBlock


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
    from data_loader.loader import DataLoader
    from sklearn.model_selection import train_test_split
    from losses.dice import DiceLoss
    X = np.random.rand(10, 160, 160, 3)
    print(X.shape)
    y = np.random.randint(0, 2, (10, 160, 160, 1))
    y = tf.keras.utils.to_categorical(y, 2)
    print(y)

    dl = DataLoader()

    dataset = dl.load_data("/Users/gonthierlucas/Desktop/repos/data/BrainTumorDataset/Mat_Format_Dataset/",
                           "mat", height=160, width=160)

    dataset = dataset.map(lambda img, mask, _ : (img, mask))
    dataset = dl.configure_for_performance(dataset, shuffle=True)
    Xs = []
    ys = []
    for _x, _y in dataset:

        Xs.append(_x)
        ys.append(_y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    y = tf.keras.utils.to_categorical(y, 2)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    print(X_train.shape)
    print(X_test.shape)
    model = U_Net((160, 160, 3), 2)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)


    loss = DiceLoss(2)
    #To test : Dice loss, BCE-Dice loss, Jaccard loss, Focal loss
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.AUC(), tf.keras.metrics.Recall()])


    epochs = 5
    model.fit(X_train, y_train, epochs=epochs, batch_size=8, validation_data=(X_test, y_test))

    print(X_test.shape)
    j = 1
    for i in range(0, 20, 2):
        if ((i+1) == 11) or ((i+2)==1):
            break
        y_pred = model.predict(np.expand_dims(X_test[i], 0))

        fig = plt.figure(figsize=(14, 14))
        fig.add_subplot(5, 2, i+1)
        plt.imshow(y_pred[0][:,:,0])
        plt.axis('off')
        plt.title("Pred")

        fig.add_subplot(5, 2, i+2 )
        plt.imshow(y_test[i][:,:,0])
        plt.axis('off')
        plt.title("gt")
        j += 2

    plt.show()
