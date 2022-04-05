import tensorflow as tf
import numpy as np
from data_loader.loader import DataLoader
from models.unet import UNet
from models.vgg import VGG
from sklearn.model_selection import train_test_split
from ops.transition import get_segmented_part, extract_roi_from_img, get_tensor_from_dataset


if __name__=="__main__":
    batch_sizes = [8, 8]
    epochs = [8, 8]
    end_to_end = True

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

    # Categorized mask
    mask = tf.keras.utils.to_categorical(mask, 2)

    # Split data
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, indices, test_size=.2)
    mask_train, label_train = mask[y_train], labels[y_train]
    mask_test, label_test = mask[y_test], labels[y_test]

    assert len(X_train) == len(mask_train)


    # Segmentation Training
    print("Begin segmentation\n\n")
    model = UNet((160, 160, 3), 2)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.AUC(), tf.keras.metrics.Recall()])

   
    model.fit(X_train, mask_train, epochs=epochs[0], batch_size=batch_sizes[0], validation_data=(X_test, mask_test))

    # Segmentation Prediction
    print("\n\nBegin segmentation prediction\n\n")

    if end_to_end:
      mask_train = tf.expand_dims(tf.math.argmax(model.predict(X_train), axis=-1), axis=-1)
    else:
      mask_train = tf.expand_dims(tf.math.argmax(mask_train, axis=-1), axis=-1)

    mask_test_pred = tf.expand_dims(tf.math.argmax(model.predict(X_test), axis=-1), axis=-1)
    #mask_test = tf.math.argmax(mask_test, axis=-1)


    # Processing Pre Classification
    print(("\n\nBegin process train data \n\n"))

    assert len(X_train) == len(mask_train)
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, mask_train, label_train))
    ds_train = ds_train.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
    ds_train = ds_train.map(lambda img, label: (extract_roi_from_img(img, 0, 224, 224), label))

    print(("\n\nBegin process test data \n\n"))
    assert len(X_test) == len(mask_test_pred)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, mask_test_pred, label_test))
    ds_test = ds_test.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
    ds_test = ds_test.map(lambda img, label: (extract_roi_from_img(img, 0, 224, 224), label))

    print("\n\nTransform to numpy\n\n")
    ds_train = dl.configure_for_performance(ds_train, shuffle=True)
    ds_test = dl.configure_for_performance(ds_test, shuffle=True)

    crop_img_train, label_train = get_tensor_from_dataset(ds_train)
    crop_img_test, label_test = get_tensor_from_dataset(ds_test)

    # Classification
    print("Begin classification\n\n")

    model = VGG((224, 224, 3))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    vgg = model.build_graph()
    
    vgg.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    vgg.fit(crop_img_train, label_train, validation_data=(crop_img_test, label_test), batch_size=batch_sizes[1], epochs=epochs[1], verbose=1)





    