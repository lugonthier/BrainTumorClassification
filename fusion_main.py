import tensorflow as tf
import numpy as np
from data_loader.loader import DataLoader
from models.unet import UNet
from models.vgg import VGG
from ops.transition import get_segmented_part, extract_roi_from_img, roi_mask_augmentation
from utils import  get_tensor_from_2D_dataset, get_tensor_from_3D_dataset
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    batch_sizes = [4, 4]
    epochs = [3, 4]
    end_to_end = False
    roi_augmentation = False
    data_augmentation = False
    train = True
    predict_test = True
    # Loading data
    print("Loading data\n\n")

    


    dl = DataLoader()
    train_dataset = dl.load_data("/content/dataset/train/", "mat", height=160, width=160)
    valid_dataset = dl.load_data("/content/dataset/Validation/", "mat", height=160, width=160)
   
    train_dataset = dl.configure_for_performance(train_dataset, shuffle=True)
    valid_dataset = dl.configure_for_performance(valid_dataset, shuffle=True)
  
    print("Convert to numpy\n\n")
    
    img_train, mask_train, label_train = get_tensor_from_3D_dataset(train_dataset)
    img_valid, mask_valid, label_valid = get_tensor_from_3D_dataset(valid_dataset)

    img_train, mask_train, label_train = img_train[0:100], mask_train[0:100], label_train[0:100]
    img_valid, mask_valid, label_valid =img_valid[0:10], mask_valid[0:10], label_valid[0:10]
    # Categorized mask
    mask_train = tf.keras.utils.to_categorical(mask_train, 2)
    mask_valid = tf.keras.utils.to_categorical(mask_valid, 2)




    # Segmentation Training
    print("Begin segmentation\n\n")
    model = UNet((160, 160, 3), 2)

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    model.fit(img_train, mask_train, epochs=epochs[0], batch_size=batch_sizes[0], validation_data=(img_valid, mask_valid))

    # Segmentation Prediction
    print("\n\nBegin segmentation prediction\n\n")

    if end_to_end:
      mask_train = tf.expand_dims(tf.math.argmax(model.predict(img_train), axis=-1), axis=-1)
    else:
      mask_train = tf.expand_dims(tf.math.argmax(mask_train, axis=-1), axis=-1)

    mask_valid_pred = tf.expand_dims(tf.math.argmax(model.predict(img_valid), axis=-1), axis=-1)
   
    
    # Processing Pre Classification

    # Build tensorflow dataset to process.
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, mask_train, label_train))
    ds_valid = tf.data.Dataset.from_tensor_slices((img_valid, mask_valid_pred, label_valid))
  
    # ROI augmentation
    if roi_augmentation:
        print("\n\n ROI Augmentation \n\n")
        ds_train = ds_train.map(lambda img, mask, label: (img, tf.map_fn(roi_mask_augmentation, mask, dtype=tf.float32), label))
        ds_valid = ds_valid.map(lambda img, mask, label: (img, tf.map_fn(roi_mask_augmentation, mask, dtype=tf.float32), label))


    print(("\n\nBegin process train data \n\n"))
    ds_train = ds_train.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
    
    ds_train = ds_train.map(lambda img, label: (extract_roi_from_img(img, 0, 224, 224), label))

    print(("\n\nBegin process valid data \n\n"))
    ds_valid = ds_valid.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
    ds_valid = ds_valid.map(lambda img, label: (extract_roi_from_img(img, 0, 224, 224), label))
   

    print("\n\nTransform to numpy\n\n")
    ds_train = dl.configure_for_performance(ds_train, shuffle=True)
    ds_valid = dl.configure_for_performance(ds_valid, shuffle=True)

    crop_img_train, label_train = get_tensor_from_2D_dataset(ds_train)
    crop_img_valid, label_valid = get_tensor_from_2D_dataset(ds_valid)
    
    # Classification
    print("Begin classification\n\n")
    model = VGG((224, 224, 3))
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    vgg = model.build_graph()
    
    vgg.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    vgg.fit(crop_img_train, label_train, validation_data=(crop_img_valid, label_valid), batch_size=batch_sizes[1], epochs=epochs[1], verbose=1)

    if predict_test:
        test_dataset = dl.load_data("/content/dataset/Test/", "mat", height=160, width=160)
        test_dataset = dl.configure_for_performance(test_dataset, shuffle=True)

        img_test, mask_test, label_test = get_tensor_from_3D_dataset(test_dataset)
        

        mask_pred = tf.expand_dims(tf.math.argmax(model.predict(img_test), axis=-1), axis=-1)
        img_seg = get_segmented_part(tf.cast(img_test, tf.float32), mask_pred)
        

        roi = tf.expand_dims(extract_roi_from_img(img_seg[0], 0, 224, 224), axis=0)
        
        label_pred = vgg.predict(roi)

        print(label_pred)





    