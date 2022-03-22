import tensorflow as tf
from models.unet import UNet
from models.vgg import VGG

from tensorflow.keras.models import Sequential


def get_segmented_part(inputs):
    img, mask = inputs
    zeros = tf.zeros_like(mask)
    boolean_mask = tf.cast(tf.math.not_equal(mask, zeros), tf.float32)

    return tf.math.multiply(img, boolean_mask)

class Decoder(tf.keras.layers.Layer):

    def __init__(self):
        pass

    def call(self, inputs):

        #X, y = inputs
        img, mask = tf.map_fn(get_segmented_part, inputs, parallel_iterations=3)




class Classifier(tf.keras.Model):

    def __init__(self, input_shape, segmentor, classifier, decoder, name='Segment_Based_Classifier', **kwargs):
        super(Classifier, self).__init__(**kwargs)

        inputs = tf.keras.Input(shape=[input_shape])
        x = segmentor(inputs)
        x = decoder(x)
        outputs = classifier(x)

        self.model = tf.keras.Model(inputs, outputs)

    def compile(self, optimizer, losses):
        self.model.compile(loss=losses,
                           optimizer=optimizer, metrics=["accuracy"])

    def fit(self, X, Y, epochs, batch):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch, validation_split=0.1)

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)

    def predict(self, X):
        return self.model.predict(X)
