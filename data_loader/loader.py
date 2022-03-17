import tensorflow as tf
import pathlib
import numpy as np
from data_loader.utils import decode_img
from pymatreader import read_mat
import h5py

import cv2
class DataLoader:

    def __init__(self):
        pass

    def load_data(self, path, file_type, mask=True,
                  label=False, height=None, width=None ):
        """Loads dataset.
        Arguments:
           path: Path to dataset.
           file_type : string `png` and `mat`.
           mask : boolean to specify if mask should be return.
           label : boolean to specify if label should be return.

        Return : A tensorflow dataset.
        """

        data_dir = pathlib.Path(path)
        input = []
        # If we want to resize.
        if (not (height is None)) and (not (height is None)):
            input.append(tf.cast(height, tf.int32))
            input.append(tf.cast(width, tf.int32))

        # Test which format is used.
        if file_type == 'png':
            folders = ['1', '2', '3']
            dataset_shape = [tf.float32, tf.int32]
            func = self.process_png
        elif file_type == 'mat':
            root = 'brainTumorDataPublic_part'
            folders = [root+'1', root+'2', root+'3', root+'4']
            dataset_shape = [tf.float32, tf.float32, tf.int32]
            func = self.process_mat
        else:
            print(f'file type : {file_type} unknown.')
            raise Exception

        datasets = []
        for folder in folders:
            imgs = tf.data.Dataset.list_files(str(data_dir / (folder + '/*')), shuffle=False)
            datasets.append(imgs.map(
                lambda x: tf.py_function(func=func, inp=[x]+input, Tout=dataset_shape),
                num_parallel_calls=tf.data.AUTOTUNE))

        ds = datasets[0].concatenate(datasets[1])
        dataset = ds.concatenate(datasets[2])

        return dataset


    def process_png(self, file_path, height=None, width=None):
        """
        (height, width) : vgg -> (224, 224), Unet -> (160, 160)
        """
        folder = int(tf.strings.split(file_path, '/').numpy()[-2].decode('utf-8'))
        one_hot = np.array([0]*3)
        one_hot[folder-1] += 1

        img = tf.io.read_file(file_path)
        img = decode_img(img, 'png', height, width)

        tf_img = tf.cast(img, tf.float32) / 255.0
        tf_one_hot_label = tf.constant(one_hot, dtype=tf.int32)
        return tf_img, tf_one_hot_label

    def process_mat(self, file_path,  height=None, width=None):
        if tf.is_tensor(height) and tf.is_tensor(width):
            height = height.numpy()
            width = width.numpy()
        #mat_file = loadmat(file_path.numpy().decode('utf-8'))['cjdata']
        with h5py.File(file_path.numpy().decode('utf-8'), 'r') as f:
            img = np.array(f.get('cjdata/image')).astype(np.float64)
            mask = np.array(f.get('cjdata/tumorMask'))
            label = np.array(f.get('cjdata/label'))
            f.close()

        hi = np.max(img)
        lo = np.min(img)
        img = (((img - lo) / (hi - lo)) * 255).astype(np.uint8)

        img = tf.expand_dims(tf.cast(img, tf.float32), -1)
        x1 = img.shape[0]
        x2 = img.shape[1]
        img = tf.reshape(tf.broadcast_to(img, (x1, x2, 3)), (x1, x2, 3))

        mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        img = decode_img(img, img_height=height, img_width=width)
        mask = decode_img(mask, img_height=height, img_width=width)


        one_hot_label = np.array([0]*3)
        one_hot_label[int(label)-1] += 1

        return img, mask, tf.constant(one_hot_label, dtype=tf.int32)


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    dl = DataLoader()

    dataset = dl.load_data("/Users/gonthierlucas/Desktop/repos/data/BrainTumorDataset/Mat_Format_Dataset",
                           "mat")

    for X, y,_ in dataset:

        print(tf.shape(X))
        plt.imshow(X.numpy().astype(np.uint8))
        plt.show()

        break