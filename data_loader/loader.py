import tensorflow as tf
import pathlib
import numpy as np
from data_loader.utils import decode_img


class DataLoader:

    def __init__(self):
        pass


    def load_data(self, path):
        """Loads dataset.
        Arguments:
           path: Path to dataset.

        Return : A tensorflow dataset.
        """

        data_dir = pathlib.Path(path)
        folders = ['1', '2', '3']
        datasets = []
        for folder in folders:
            imgs = tf.data.Dataset.list_files(str(data_dir / (folder + '/*')), shuffle=False)
            datasets.append(imgs.map(
                lambda x: tf.py_function(func=self.process, inp=[x], Tout=[tf.float32, tf.int32]),
                num_parallel_calls=tf.data.AUTOTUNE))

        ds = datasets[0].concatenate(datasets[1])
        dataset = ds.concatenate(datasets[2])

        return dataset


    def process(self, file_path):
        folder = int(tf.strings.split(file_path, '/').numpy()[-2].decode('utf-8'))
        one_hot = np.array([0]*3)
        one_hot[folder-1] += 1

        img = tf.io.read_file(file_path)
        img = decode_img(img, 224, 224)

        tf_img = tf.cast(img, tf.float32) / 255.0
        tf_one_hot_label = tf.constant(one_hot, dtype=tf.int32)
        return tf_img, tf_one_hot_label

    def configure_for_performance(self, dataset, shuffle: bool = False, batch_size: int =32,
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

