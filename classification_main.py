import numpy as np
import tensorflow as tf

from data_loader.loader import DataLoader
from models.vgg import VGG
if __name__=="__main__":


    print(tf.__version__)

    loader = DataLoader()
    ds = loader.load_data('BrainTumorDataset', 'png')
    ds = loader.configure_for_performance(ds, shuffle=True)

    model = VGG((224, 224, 3))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    vgg = model.build_graph()

    Xs = []
    ys = []
    for _x, _y in ds:
        Xs.append(_x)
        ys.append(_y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    vgg.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


    vgg.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=8, epochs=10, verbose=1)

    print(vgg.summary())




