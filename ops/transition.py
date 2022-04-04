import numpy as np
import tensorflow as tf

from scipy import ndimage

def get_tensor_from_dataset(dataset):
    """From a tensorflow dataset, return a list of tensor.
    """

    Xs = []
    ys = []
    for x, y in dataset:
        Xs.append(x)
        ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def get_segmented_part(img, mask):
    """From the image conserve only part corresponding to the ROI. The rest will be set to 0.
    """
    zeros = tf.zeros_like(mask)
    boolean_mask = tf.cast(tf.math.not_equal(mask, zeros), tf.float32)

    return tf.math.multiply(img, boolean_mask)


def roi_mask_augmentation(mask, h=40, w=40, center=None, radius=None):
    
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    circular_mask = dist_from_center <= radius
    return ndimage.binary_dilation(mask, circular_mask)


def extract_roi_from_img(image, tol=0, height=None, width=None):
    """ Extract non zeros part of an image.
    Used to crop an image to get only the region of interest.
    """

    tol = tf.cast(tol,tf.float32)
    shape = tf.shape(image)
    m, n = shape[0], shape[1]

    channels = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    min_col_start = n
    max_col_end = 0
    min_row_start = m
    max_row_end = 0
    # For each channel
    for ind in tf.range(tf.shape(image)[-1]):
        # Because python lists not work well with TF control flow.
        # Cf: https://github.com/tensorflow/tensorflow/issues/37512

        img = image[:,:,ind]

        # Get the mask based on tol value.
        mask = tf.cast(tf.greater(img, tf.fill(tf.shape(img), tol)), tf.float32)

        mask0, mask1 = tf.experimental.numpy.any(mask, 0), tf.experimental.numpy.any(mask, 1)
        
        # Get indices to crop the image.
        col_start, col_end = tf.argmax(mask0, output_type=tf.int32), tf.cast(n, tf.int32) - tf.argmax(mask0[::-1], output_type=tf.int32)
        row_start, row_end = tf.argmax(mask1, output_type=tf.int32), tf.cast(m, tf.int32) - tf.argmax(mask1[::-1], output_type=tf.int32)
    
        if col_start < min_col_start:
          min_col_start = col_start
        if col_end > max_col_end:
          max_col_end = col_end
        if row_start < min_row_start:
          min_row_start = row_start
        if row_end > max_row_end:
          max_row_end = row_end
        
    for ind in tf.range(tf.shape(image)[-1]):
      img = image[:,:,ind]
      new = img[min_row_start:max_row_end, min_col_start:max_col_end]
          
      channels = channels.write(channels.size(), new)

    # Reshape with channels at the end.
    channels = channels.stack()
    shape = tf.shape(channels)
    new_image = tf.reshape(channels, [shape[1], shape[2], shape[0]])
 
    if (height is not None) and (width is not None):
        new_image = tf.image.resize(new_image, [height, width])
    
    return new_image