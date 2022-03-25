import tensorflow as tf


def configure_for_performance(dataset, shuffle: bool = False, batch_size: int =32,
                                  buffer_size: int = 1000):
        """
        Configure dataset for performance.
        https://www.tensorflow.org/guide/performance/datasets
        """
        dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

def decode_img(img, format=None, img_height: int = None, img_width: int = None):
    # Convert the compressed string to a 3D uint8 tensor
    if tf.is_tensor(img_height) and tf.is_tensor(img_width):
        img_height = img_height.numpy()
        img_width = img_width.numpy()
    if format =='png':
        img = tf.io.decode_jpeg(img, channels=3)

    if (img_height and img_width):
        # Resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    else:
        # No Resizing.
        return img