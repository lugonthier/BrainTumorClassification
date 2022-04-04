import tensorflow as tf
from models.unet import UNet
from models.vgg import VGG
from tensorflow.keras.models import Sequential
from data_loader.utils import configure_for_performance
from ops.transition import get_segmented_part, extract_roi_from_img, get_tensor_from_dataset

class Encoder(tf.keras.layers.Layer):

    def __init__(self):
        pass

    def call(self, masks):

        #X, y = inputs
        img, mask = tf.map_fn(get_segmented_part, inputs, parallel_iterations=3)




class Classifier(tf.keras.Model):

    def __init__(self, input_shapes, segmentor, classifier, name='Segment_Based_Classifier', **kwargs):
        super(Classifier, self).__init__(**kwargs)

        self.segmentor_input_shape, self.classifier_input_shape = input_shapes
        self.segmentor = segmentor
        self.classifier = classifier


    def compile(self, optimizers, losses, metrics):
        # Compile segmentor module
        self.segmentor.compile(loss=losses[0],
                           optimizer=optimizers[0], metrics=metrics[0])

        # Compile classifier module
        self.classifier.compile(loss=losses[1],
                           optimizer=optimizers[1], metrics=metrics[1])


    def fit(self, train_inputs, valid_inputs, epochs=[1, 1], batch_size=[8, 8], end_to_end=False):
       
        train_img, train_mask, train_label = train_inputs
        valid_img, valid_mask, valid_label = valid_inputs
        
        self.segmentor.fit(train_img, train_mask, epochs=epochs[0], batch_size=batch_size[0], validation_data=(valid_img, valid_mask))

        if (end_to_end):
            
            train_mask = tf.expand_dims(tf.math.argmax(self.segmentor.predict(train_img), axis=-1), axis=-1)

        # Remove categorical shape.
        valid_mask_pred = tf.expand_dims(tf.math.argmax(self.segmentor.predict(valid_img), axis=-1), axis=-1)
        train_mask = tf.expand_dims(tf.math.argmax(train_mask, axis=-1), axis=-1)
        valid_mask = tf.math.argmax(valid_mask, axis=-1)
        
        ds_train = tf.data.Dataset.from_tensor_slices((train_img, train_mask, train_label))
        ds_valid = tf.data.Dataset.from_tensor_slices((valid_img, valid_mask_pred, valid_label))

        # Get segmented part from img.
        ds_train = ds_train.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))
        ds_valid = ds_valid.map(lambda img, mask, label: (get_segmented_part(tf.cast(img, tf.float32), mask), label))

        # Extract ROI from img and resize it.
        ds_train = ds_train.map(lambda img, label: (extract_roi_from_img(img, 0, self.classifier_input_shape, self.classifier_input_shape), label))
        ds_valid = ds_valid.map(lambda img, label: (extract_roi_from_img(img, 0, self.classifier_input_shape, self.classifier_input_shape), label))

        ds_train = configure_for_performance(ds_train, shuffle=True)
        ds_valid = configure_for_performance(ds_valid, shuffle=True)
        train_img, train_label = get_tensor_from_dataset(ds_train)
        valid_img, valid_label = get_tensor_from_dataset(ds_valid)
        
        # TODO: Roi augmentation

        self.classifier.fit(train_img, train_label, epochs=epochs[1], batch_size=batch_size[0], validation_data=(valid_img, valid_label))
        

    def evaluate(self, inputs):
        
         

        masks = self.segmentor.predict(imgs)

        # Encode masks
        # Roi augmentation

        self.classifier.evaluate(masks, labels)

        

    def predict(self, img, return_mask=False):
        
        mask_pred = tf.expand_dims(tf.math.argmax(self.segmentor.predict(img), axis=-1), axis=-1)
        img_seg = get_segmented_part(tf.cast(img, tf.float32), mask_pred)
        

        roi = tf.expand_dims(extract_roi_from_img(img_seg[0], 0, self.classifier_input_shape, self.classifier_input_shape), axis=0)
        
        label_pred = self.classifier.predict(roi)

        if return_mask:
            return (mask_pred, label_pred)

        return label_pred


if __name__=="__main__":
    import numpy as np
    from data_loader.loader import DataLoader
    from models.unet import UNet
    from models.vgg import VGG
    from sklearn.model_selection import train_test_split


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


