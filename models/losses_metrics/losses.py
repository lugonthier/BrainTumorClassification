import tensorflow as tf



def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = tf.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss