import tensorflow as tf
import numpy as np
from data_loader.loader import DataLoader
from models.unet import UNet
from models.vgg import VGG
from sklearn.model_selection import train_test_split

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

    # Loading data
    print("Loading data\n\n")
    dl = DataLoader()
    dataset = dl.load_data("/content/BrainTumorDataset/Mat_Format_Dataset/", "mat", height=160, width=160)

    print("Convert to numpy\n\n")
    dataset = dl.configure_for_performance(dataset, shuffle=True)
    Xs = []
    masks = []
    labels = []
    for _x, _mask, label in dataset:

        Xs.append(_x)
        masks.append(_mask)
        labels.append(label)

    X = np.concatenate(Xs, axis=0)
    print(X.shape)
    mask = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)

    mask = tf.keras.utils.to_categorical(mask, 2)

    indices = np.arange(len(X))
    
    X_train, X_test, y_train, y_test = train_test_split(X, indices, test_size=.2)

    mask_train, label_train = mask[y_train], labels[y_train]
    mask_test, label_test = mask[y_test], labels[y_test]

    assert len(X_train) == len(mask_train)


    # Begin segmentation
    print("Begin segmentation\n\n")
    model = UNet((160, 160, 3), 2)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.AUC(), tf.keras.metrics.Recall()])

    epochs = 1
    model.fit(X_train, mask_train, epochs=epochs, batch_size=8, validation_data=(X_test, mask_test))

    print("\n\nBegin segmentation prediction\n\n")
    mask_train_pred = tf.expand_dims(tf.math.argmax(model.predict(X_train), axis=-1), axis=-1)
    mask_test_pred = tf.expand_dims(tf.math.argmax(model.predict(X_test), axis=-1), axis=-1)

    mask_train = tf.expand_dims(tf.math.argmax(mask_train, axis=-1), axis=-1)
    mask_test = tf.math.argmax(mask_test, axis=-1)


    print(("\n\nBegin process train data \n\n"))

    assert len(X_train) == len(mask_train_pred)
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, mask_train, label_train))
    print(ds_train)
    ds_train = ds_train.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
    ds_train = ds_train.map(lambda img, label: (tf.py_function(crop_image_padding, [img, 0, 224, 224], [tf.float32]), label))

    print(("\n\nBegin process test data \n\n"))
    assert len(X_test) == len(mask_test_pred)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, mask_test_pred, label_test))
    ds_test = ds_test.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
    ds_test = ds_test.map(lambda img, label: (tf.py_function(crop_image_padding, [img, 0, 224, 224], [tf.float32]), label))

    print("\n\nTransform to numpy\n\n")
    ds_train = dl.configure_for_performance(ds_train, shuffle=True)

    Xs = []
    ys = []
    for _x, _y in ds_train:

        Xs.append(tf.squeeze(_x, axis=1))
        ys.append(_y)
    X_train = np.concatenate(Xs, axis=0)
    y_train = np.concatenate(ys, axis=0)

    ds_test = dl.configure_for_performance(ds_test, shuffle=True)
    Xs = []
    ys = []
    for _x, _y in ds_test:

        Xs.append(tf.squeeze(_x, axis=1))
        ys.append(_y)
    X_test = np.concatenate(Xs, axis=0)
    y_test = np.concatenate(ys, axis=0)

    print("Begin classification\n\n")

    model = VGG((224, 224, 3))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    vgg = model.build_graph()

    
    vgg.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    vgg.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=8, epochs=1, verbose=1)





    