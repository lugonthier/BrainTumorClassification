import numpy as np
import tensorflow as tf
from data_loader.loader import DataLoader
from models.unet import UNet
from models.vgg import VGG
from models.classifier import Classifier
from sklearn.model_selection import train_test_split



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

    #models
    input_shapes = [160, 224]
    segmentor = UNet((160, 160, 3), 2)
    classifier = VGG((224, 224, 3)).build_graph()

    model = Classifier(input_shapes, segmentor, classifier)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizers=["rmsprop", "rmsprop"], losses=["binary_crossentropy", "categorical_crossentropy"], metrics=[[tf.keras.metrics.MeanIoU(num_classes=2)], ["accuracy"]])
    model.fit(train_inputs=[X_train, mask_train, label_train], valid_inputs=[X_test, mask_test, label_test], end_to_end=True)

    x = np.expand_dims(X_test[0], axis=0)
    prediction = model.predict(x, True)
    print(prediction)