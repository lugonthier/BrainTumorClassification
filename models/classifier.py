import tensorflow as tf
from models.unet import UNet
from models.vgg import VGG

from tensorflow.keras.models import Sequential


def get_segmented_part(inputs):
    img, mask = inputs
    zeros = tf.zeros_like(mask)
    boolean_mask = tf.cast(tf.math.not_equal(mask, zeros), tf.float32)

    return tf.math.multiply(img, boolean_mask)

class Encoder(tf.keras.layers.Layer):

    def __init__(self):
        pass

    def call(self, masks):

        #X, y = inputs
        img, mask = tf.map_fn(get_segmented_part, inputs, parallel_iterations=3)




class Classifier(tf.keras.Model):

    def __init__(self, input_shape, segmentor, classifier, encoder, name='Segment_Based_Classifier', **kwargs):
        super(Classifier, self).__init__(**kwargs)

        self.segmentor = segmentor
        self.classifier = classifier
        self.encoder

    def compile(self, optimizers, losses, metrics):
        # Compile segmentor module
        self.segmentor.compile(loss=losses[0],
                           optimizer=optimizer[0], metrics=metrics[0])

        # Compile classifier module
        self.classifier.compile(loss=losses[1],
                           optimizer=optimizer[1], metrics=metrics[1])

    def fit(self, inputs, end_to_end=False):
        imgs, masks, labels = inputs

        self.segmentor.fit(imgs, masks)

        if (end_to_end):
            masks = self.segmentor.predict(imgs)
        
        # Encode masks 
        # Roi augmentation

        self.classifier.fit(masks, labels)
        

    def evaluate(self, inputs):
        imgs, masks, labels = inputs

        masks = self.segmentor.predict(imgs)

        # Encode masks
        # Roi augmentation

        self.classifier.evaluate(masks, labels)

        

    def predict(self, X):
        
        masks = self.segmentor.predict(X)

        # encode masks
        # Roi augmentation

        return self.classifier.predict(masks)
