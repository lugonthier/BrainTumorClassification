from operator import ne
import tensorflow as tf
import numpy as np
from data_loader.loader import DataLoader
from sklearn.model_selection import train_test_split
from models.unet import UNet
from models.vgg import VGG

def get_segmented_part(img, mask):
    zeros = tf.zeros_like(mask)
    boolean_mask = tf.cast(tf.math.not_equal(mask, zeros), tf.float32)

    return tf.math.multiply(img, boolean_mask)


def crop_image_padding(image, tol=0, height=None, width=None):
    # img is 3D image data with transparency channel
    # tol is tolerance (0 in your case)
    # we are interested only in the transparency layer so:
  
    image = image.numpy()
    tol = tol.numpy()
    channels = []
    for ind in range(3):
      img = image[:,:,ind]
      mask = img > tol
      shape = img.shape
      m, n = shape[0], shape[1]
      mask0, mask1 = mask.any(0), mask.any(1)
      col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
      row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
      
      new = tf.cast(img[row_start:row_end, col_start:col_end], tf.int32)
      
      channels.append(tf.cast(new, tf.int32))

    channels = tf.transpose(channels, [1, 2, 0])
    if (height is not None) and (width is not None):
        new_image = tf.image.resize(channels, [height, width])
    else:
      tf.cast(channels, tf.int32)

   
    return new_image

if __name__=="__main__":
    dl = DataLoader()
    #Users/gonthierlucas/Desktop/repos/data
    dataset = dl.load_data("/content/BrainTumorDataset/Mat_Format_Dataset/",
                           "mat", height=224, width=224)

    ds = dataset.map(lambda img, mask, label: (get_segmented_part(img, mask), label))
    ds = ds.map(lambda img, label: (tf.py_function(crop_image_padding, [img, 0, 224, 224], [tf.float32]  ), label))


    ds = dl.configure_for_performance(ds, shuffle=True)

    Xs = []
    ys = []
    for _x, _y in ds:

        Xs.append(tf.squeeze(_x, axis=1))
        ys.append(_y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    # Model
    print(X.shape)
    model = VGG((224, 224, 3))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    vgg = model.build_graph()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    vgg.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    vgg.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=8, epochs=10, verbose=1)