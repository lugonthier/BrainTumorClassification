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

if __name__=="__main__":
    dl = DataLoader()

    dataset = dl.load_data("/Users/gonthierlucas/Desktop/repos/data/BrainTumorDataset/Mat_Format_Dataset/",
                           "mat", height=224, width=224)

    ds = dataset.map(lambda img, mask, label: (get_segmented_part(img, mask), label))
    ds = dl.configure_for_performance(ds, shuffle=True)

    Xs = []
    ys = []
    for _x, _y in ds:
        Xs.append(_x)
        ys.append(_y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    # Model
    model = VGG((224, 224, 3))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    vgg = model.build_graph()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    vgg.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    vgg.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=8, epochs=10, verbose=1)