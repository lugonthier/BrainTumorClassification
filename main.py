

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from models.vgg import VGG
from models.unet import UNet
from data_loader.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import  get_tensor_from_2D_dataset, get_tensor_from_3D_dataset
from ops.transition import get_segmented_part, extract_roi_from_img, roi_mask_augmentation


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with
        datasets and differents parameters.
    """
    parser = argparse.ArgumentParser(usage='\n python3 main.py  [dataset] [hyper_parameters]',
                                     description="")

    parser.add_argument('--train_set', type=str, default='train_equalize', choices = ['train_original', 'train_equalize', 'train_augmente'])
    parser.add_argument('--segmentor_batch_size', type=int, default=2,
                        help='The size of the training batch for the segmentor')
    
    parser.add_argument('--classifier_batch_size', type=int, default=2,
                        help='The size of the training batch for the classifier')
    
    parser.add_argument('--segmentor_epochs', type=int, default=2,
                        help='Number of epochs for the segmentor')
    
    parser.add_argument('--classifier_epochs', type=int, default=2,
                        help='Number of epochs for the classifier')
    
    parser.add_argument('--end_to_end', type=bool, default=False,
                        help='To perform end to end training.')

    parser.add_argument('--roi_augmentation', type=bool, default=False,
                        help='To perform roi augmentation')

    parser.add_argument('--predict_test', type=bool, default=True,
                        help='To perform data augmentation')

    return parser.parse_args()
    
    

if __name__=="__main__":
    args = argument_parser()

    batch_sizes = [args.segmentor_batch_size, args.classifier_batch_size]
    epochs = [args.segmentor_epochs  , args.classifier_epochs]
    end_to_end = args.end_to_end
    roi_augmentation = args.roi_augmentation
    train_set = args.train_set
    predict_test = args.predict_test

    experimentation_name = f'experiment__{train_set}_batch_sizes=({batch_sizes[0]},{batch_sizes[1]})_epochs=({epochs[0]},{epochs[1]})'
    if end_to_end:
        experimentation_name += '_end_to_end'
    
    if roi_augmentation:
      experimentation_name += '_roi_augmentation'
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    path = f'results/{experimentation_name}'
    if not os.path.exists(path):
        os.makedirs(path)


    # Loading data
    print("Loading data...\n")

    dl = DataLoader()
    train_dataset = dl.load_data(f"/content/dataset/{train_set}/", "mat", height=160, width=160)
    valid_dataset = dl.load_data("/content/dataset/Validation/", "mat", height=160, width=160)
   
    train_dataset = dl.configure_for_performance(train_dataset, shuffle=True)
    valid_dataset = dl.configure_for_performance(valid_dataset, shuffle=True)
    
    # Get tensors.
    img_train, mask_train, label_train = get_tensor_from_3D_dataset(train_dataset)
    img_valid, mask_valid, label_valid = get_tensor_from_3D_dataset(valid_dataset)

    # Categorized masks.
    ind_train = len(mask_train)
    mask = np.concatenate([mask_train, mask_valid])
    mask = tf.keras.utils.to_categorical(mask, 2)
    mask_train = mask[:ind_train]
    mask_valid = mask[ind_train:]




    # Segmentation Training
    print('\n' + ('-'*10) + 'Begin segmentation'+ ('-'*10) + '\n')
    segmentor = UNet((160, 160, 3), 2)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    segmentor.compile(optimizer=opt, loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    segmentor_history = segmentor.fit(img_train, mask_train, epochs=epochs[0], batch_size=batch_sizes[0], validation_data=(img_valid, mask_valid))
    print(segmentor_history.history.keys())
    segmentor_result = pd.DataFrame(segmentor_history.history)
    segmentor_result.to_csv(f'results/{experimentation_name}/segmentor.csv')

    # Segmentation Prediction(s)
    if end_to_end:
      print("\nBegin segmentation prediction on training set\n")
      mask_train = tf.expand_dims(tf.math.argmax(segmentor.predict(img_train), axis=-1), axis=-1)
    else:
      mask_train = tf.expand_dims(tf.math.argmax(mask_train, axis=-1), axis=-1)
    print("\nSegmentation prediction on validation set\n")
    mask_valid_pred = tf.expand_dims(tf.math.argmax(segmentor.predict(img_valid), axis=-1), axis=-1)
   
    
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
    vgg_history = vgg.fit(crop_img_train, label_train, validation_data=(crop_img_valid, label_valid), batch_size=batch_sizes[1], epochs=epochs[1], verbose=1)

    print(vgg_history.history.keys())
    vgg_result = pd.DataFrame(vgg_history.history)
    vgg_result.to_csv(f'results/{experimentation_name}/vgg.csv')

    if predict_test:

        print(('-'*10) + "Prediction on Test set" + ('-'*10) + '\n')
        test_dataset = dl.load_data("/content/dataset/Test/", "mat", height=160, width=160)
        test_dataset = dl.configure_for_performance(test_dataset, shuffle=True)

        # Get tensors
        img_test, mask_test, label_test = get_tensor_from_3D_dataset(test_dataset)
        
        # Segmentation prediction
        mask_pred = tf.expand_dims(tf.math.argmax(segmentor.predict(img_test), axis=-1), axis=-1)

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
        






    