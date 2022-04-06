import tensorflow as tf
import pathlib
import numpy as np
from data_loader.utils import decode_img
from pymatreader import read_mat


class DataLoader:

    def __init__(self):
        pass

    def load_data(self, path, file_type, return_mask=True,
                  return_label=False, height=None, width=None):
        """Loads dataset.
        Arguments:
            path: Path to dataset.
            file_type : string `png` and `mat`.
            return_mask : boolean to specify if mask should be return.
            return_label : boolean to specify if label should be return.
            height : new image height wanted.
            width : new image width wanted.
        Return : A tensorflow dataset.
        """

        data_dir = pathlib.Path(path)
        inputs = []
        # If we want to resize.
        if (not (height is None)) and (not (height is None)):
            inputs.append(tf.cast(height, tf.int32))
            inputs.append(tf.cast(width, tf.int32))

        # Test which format is used.
        if file_type == 'png':
            folders = ['1/*', '2/*', '3/*']
            dataset_shape = [tf.float32, tf.int32]
            func = self.process_png

        elif file_type == 'mat':
            folders = ['*']

            dataset_shape = [tf.float32, tf.float32, tf.int32]
            func = self.process_mat
        else:
            print(f'file type : {file_type} unknown.')
            raise Exception

        datasets = []
        for folder in folders:
            imgs = tf.data.Dataset.list_files(str(data_dir / (folder)), shuffle=False)
            datasets.append(imgs.map(
                lambda x: tf.py_function(func=func, inp=[x]+inputs, Tout=dataset_shape),
                num_parallel_calls=tf.data.AUTOTUNE))

        if(file_type=='png'): 
            ds = datasets[0].concatenate(datasets[1])
            dataset = ds.concatenate(datasets[2])
        elif(file_type=='mat'):
            dataset = datasets[0]

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


        """with h5py.File(file_path.numpy().decode('utf-8'), 'r') as f:
            img = np.array(f.get('cjdata/image')).astype(np.float64)
            mask = np.array(f.get('cjdata/tumorMask'))
            label = np.array(f.get('cjdata/label'))
            f.close()"""
        
        mat = read_mat(file_path.numpy().decode('utf-8'))
        img = np.array(mat['cjdata']['image'])
        mask = np.array(mat['cjdata']['tumorMask'])
        label = np.array(mat['cjdata']['label'])

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

    dataset = dl.load_data("/Users/gonthierlucas/Desktop/repos/data/BrainTumorDataset/",
                           "png")
    for X, y in dataset:
        img = X.numpy()
        hi = np.max(img)
        lo = np.min(img)
        img = (((img - lo) / (hi - lo)) * 255).astype(np.uint8)
        print(tf.shape(X))
        plt.imshow(img)
        plt.show()

        break