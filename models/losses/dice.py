import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, n_classes,smooth=1e-6, reduction=tf.keras.losses.Reduction.AUTO, name='DiceLoss'):
        super().__init__(reduction=reduction, name=name)
        self.n_classes = n_classes
        self.smooth = smooth

    def call(self, y_true, y_pred):

        #y_pred = tf.keras.layers.Flatten('channels_first')(y_pred)
        #y_true = tf.keras.layers.Flatten('channels_first')(y_true)#tf.one_hot(tf.cast(y_true, tf.int32), depth=self.n_classes))
        print(tf.shape(y_true), tf.shape(y_pred))
        intersection = tf.reduce_sum(tf.tensordot(y_true, tf.transpose(y_pred), 1))
        dice = (2 * intersection + self.smooth) / (tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + self.smooth)

        return 1 - dice