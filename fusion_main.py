import tensorflow as tf
import numpy as np
import seaborn as sns
from data_loader.loader import DataLoader
from models.unet import UNet
from models.vgg import VGG
from ops.transition import get_segmented_part, extract_roi_from_img, roi_mask_augmentation
from utils import  get_tensor_from_2D_dataset, get_tensor_from_3D_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


if __name__=="__main__":
    batch_sizes = [2, 2]
    epochs = [10, 10]
    end_to_end = False
    roi_augmentation = True
    data_augmentation = False
    train = True
    predict_test = True
    # Loading data
    print("Loading data...\n")

    dl = DataLoader()
    train_dataset = dl.load_data("/content/dataset/train_equalize/", "mat", height=160, width=160)
    valid_dataset = dl.load_data("/content/dataset/Validation/", "mat", height=160, width=160)
   
    train_dataset = dl.configure_for_performance(train_dataset, shuffle=True)
    valid_dataset = dl.configure_for_performance(valid_dataset, shuffle=True)
    
    # Get tensors.
    img_train, mask_train, label_train = get_tensor_from_3D_dataset(train_dataset)
    img_valid, mask_valid, label_valid = get_tensor_from_3D_dataset(valid_dataset)

    # Categorized masks.
    ind_train =len(mask_train)
    mask = np.concatenate([mask_train, mask_valid])
    mask = tf.keras.utils.to_categorical(mask, 2)
    mask_train = mask[:ind_train]
    mask_valid = mask[ind_train:]




    # Segmentation Training
    print('\n' + ('-'*10) + 'Begin segmentation'+ ('-'*10) + '\n')
    model = UNet((160, 160, 3), 2)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    model.fit(img_train, mask_train, epochs=epochs[0], batch_size=batch_sizes[0], validation_data=(img_valid, mask_valid))

    # Segmentation Prediction(s)
    if end_to_end:
      print("\nBegin segmentation prediction on training set\n")
      mask_train = tf.expand_dims(tf.math.argmax(model.predict(img_train), axis=-1), axis=-1)
    else:
      mask_train = tf.expand_dims(tf.math.argmax(mask_train, axis=-1), axis=-1)
    print("\nSegmentation prediction on validation set\n")
    mask_valid_pred = tf.expand_dims(tf.math.argmax(model.predict(img_valid), axis=-1), axis=-1)
   
    
    # Processing Pre Classification
    # Build tensorflow dataset to process.
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, mask_train, label_train))
    ds_valid = tf.data.Dataset.from_tensor_slices((img_valid, mask_valid_pred, label_valid))
  
    # ROI augmentation
    if roi_augmentation:
        print("\nROI Augmentation\n")
        ds_train = ds_train.map(lambda img, mask, label: (img, tf.map_fn(roi_mask_augmentation, mask, tf.float32), label))
        ds_valid = ds_valid.map(lambda img, mask, label: (img, tf.map_fn(roi_mask_augmentation, mask, tf.float32), label))


    print("\nBegin ROI cropping process training and validation data\n")
    ds_train = ds_train.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
    ds_train = ds_train.map(lambda img, label: (extract_roi_from_img(img, 0, 224, 224), label))

    ds_valid = ds_valid.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
    ds_valid = ds_valid.map(lambda img, label: (extract_roi_from_img(img, 0, 224, 224), label))
   
    ds_train = dl.configure_for_performance(ds_train, shuffle=True)
    ds_valid = dl.configure_for_performance(ds_valid, shuffle=True)

    # Get tensors.
    crop_img_train, label_train = get_tensor_from_2D_dataset(ds_train)
    crop_img_valid, label_valid = get_tensor_from_2D_dataset(ds_valid)
    
    # Classification
    print('\n' + ('-'*10) + 'Begin classification' +('-'*10)+'\n')
    vgg = VGG((224, 224, 3))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    vgg = vgg.build_graph()
    
    vgg.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    vgg.fit(crop_img_train, label_train, validation_data=(crop_img_valid, label_valid), batch_size=batch_sizes[1], epochs=epochs[1], verbose=1)

    if predict_test:

        print(('-'*10) + "Prediction on Test set" + ('-'*10) + '\n')
        test_dataset = dl.load_data("/content/dataset/Test/", "mat", height=160, width=160)
        test_dataset = dl.configure_for_performance(test_dataset, shuffle=True)

        # Get tensors
        img_test, mask_test, label_test = get_tensor_from_3D_dataset(test_dataset)
        
        # Segmentation prediction
        mask_pred = tf.expand_dims(tf.math.argmax(model.predict(img_test), axis=-1), axis=-1)

        ds_test = tf.data.Dataset.from_tensor_slices((img_test, mask_pred, label_test))
        
        if roi_augmentation:
          print("ROI Augmentation")
          ds_test = ds_test.map(lambda img, mask, label: (img, tf.map_fn(roi_mask_augmentation, mask, tf.float32), label))

        print("Begin ROI cropping process train data")
        ds_test = ds_test.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
        ds_test = ds_test.map(lambda img, label: (extract_roi_from_img(img, 0, 224, 224), label))
        ds_test = dl.configure_for_performance(ds_test, shuffle=True)

        # Get tensors.
        crop_img_test, label_test = get_tensor_from_2D_dataset(ds_test)

        # Prediction.
        label_pred = tf.math.argmax(vgg.predict(crop_img_test), axis=-1)
        label_test = tf.math.argmax(label_test, axis=-1)

        print(f'accuracy score : {accuracy_score(label_test, label_pred)}')
        conf = confusion_matrix(label_test, label_pred)
        print(conf)
        sns.heatmap(conf, annot=True)






    