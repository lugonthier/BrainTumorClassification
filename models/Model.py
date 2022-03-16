import tensorflow as tf
import time
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input


class VGG(tf.keras.Model):
    """VGG16 model for classification."""

    def __init__(self,input_shape,  n_classes=3, name='VGG', **kwargs):
        super(VGG, self).__init__(**kwargs)

        self.in_shape=input_shape
        # B1 (Block 1)
        self.conv1 = Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), activation='relu',
                            padding='same', name='conv1')
        self.conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')
        self.maxpool1 = MaxPool2D((2, 2), strides=(2, 2), name='pool1')

        # B2
        self.conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')
        self.conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4')
        self.maxpool2 = MaxPool2D((2, 2), strides=(2, 2), name='pool2')

        # B3
        self.conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5')
        self.conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6')
        self.conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7')
        self.maxpool3 = MaxPool2D((2, 2), strides=(2, 2), name='pool3')

        # B4
        self.conv8 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv8')
        self.conv9 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv9')
        self.conv10 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv10')
        self.maxpool4 = MaxPool2D((2, 2), strides=(2, 2), name='pool4')

        # B5
        self.conv11 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv11')
        self.conv12 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv12')
        self.conv13 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv13')
        self.maxpool5 = MaxPool2D((2, 2), strides=(2, 2), name='pool5')

        # FC
        self.flatten = Flatten()
        self.dense1 = Dense(units=4096, activation="relu")
        self.dense2 = Dense(units=4096, activation="relu")
        self.output_layer = Dense(units=n_classes, activation="softmax")



    def call(self, inputs, training=False):
        #inputs = input(inputs)
        # B1
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool1(x)
        # B2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        # B3
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool3(x)
        # B4
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool4(x)

        # B5
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxpool5(x)

        # FC
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)

        return x

    """
    def train_step(self, x, y):
        '''
        input: x, y <- typically batches
        input: step <- batch step
        return: loss value
        '''

        # start the scope of gradient
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # compute gradient
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value

            return {m.name: m.result() for m in self.metrics}

    def test_step(self, x, y):
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    def fit(self):
        for epoch in range(epochs):
            t = time.time()
            # batch training

            # Iterate over the batches of the train dataset.
            for train_batch_step, (x_batch_train, \
                                   y_batch_train) in enumerate(train_dataset):
                train_batch_step = tf.convert_to_tensor(train_batch_step, dtype=tf.int64)
                train_loss_value = train_step(train_batch_step,
                                              x_batch_train, y_batch_train)

            # evaluation on validation set
            # Run a validation loop at the end of each epoch.
            for test_batch_step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                test_batch_step = tf.convert_to_tensor(test_batch_step, dtype=tf.int64)
                val_loss_value = test_step(test_batch_step, x_batch_val, y_batch_val)

            template = 'ETA: {} - epoch: {} loss: {}  acc: {} val loss: {} val acc: {}\n'
            print(template.format(
                round((time.time() - t) / 60, 2), epoch + 1,
                train_loss_value, float(train_acc_metric.result()),
                val_loss_value, float(val_acc_metric.result())
            ))

            # Reset metrics at the end of each epoch
            train_acc_metric.reset_states()
            val_acc_metric.reset_states()"""
    def build_graph(self):
        x = Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))




if __name__ == "__main__":
    import numpy as np
    from data_loader.loader import DataLoader

    print(tf.__version__)

    loader = DataLoader()
    ds = loader.load_data('/Users/gonthierlucas/Desktop/repos/data/BrainTumorDataset')
    ds = loader.configure_for_performance(ds, shuffle=True)


    model = VGG((224, 224, 3))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    vgg= model.build_graph()

    Xs = []
    ys = []
    for _x, _y in ds:
        Xs.append(_x)
        ys.append(_y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    vgg.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


    vgg.fit(X, y,  batch_size=32, epochs=1, verbose=1)

    print(vgg.summary())




