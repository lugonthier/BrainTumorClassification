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

    epochs = 12
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
    ds_train = ds_train.map(lambda img, label: (extract_roi_from_img(img, 0, 224, 224), label))

    print(("\n\nBegin process test data \n\n"))
    assert len(X_test) == len(mask_test_pred)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, mask_test_pred, label_test))
    ds_test = ds_test.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
    ds_test = ds_test.map(lambda img, label: (extract_roi_from_img(img, 0, 224, 224), label))

    print("\n\nTransform to numpy\n\n")
    ds_train = dl.configure_for_performance(ds_train, shuffle=True)

    Xs = []
    ys = []
    for _x, _y in ds_train:
     
        Xs.append(_x)
        ys.append(_y)
    X_train = np.concatenate(Xs, axis=0)
    y_train = np.concatenate(ys, axis=0)

    ds_test = dl.configure_for_performance(ds_test, shuffle=True)
    Xs = []
    ys = []
    for _x, _y in ds_test:

        Xs.append(_x)
        ys.append(_y)
    X_test = np.concatenate(Xs, axis=0)
    y_test = np.concatenate(ys, axis=0)

    print("Begin classification\n\n")

    model = VGG((224, 224, 3))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    vgg = model.build_graph()

    
    vgg.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    vgg.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=8, epochs=12, verbose=1)





    