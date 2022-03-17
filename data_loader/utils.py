import tensorflow as tf
def decode_img(img, format=None, img_height: int = None, img_width: int = None):
    # Convert the compressed string to a 3D uint8 tensor

    if format =='png':
        img = tf.io.decode_jpeg(img, channels=3)

    if (img_height and img_width):
        # Resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    else:
        # No Resizing.
        return img