import numpy as np
import tensorflow as tf
from data_loader.loader import DataLoader
from models.unet import UNet
from models.vgg import VGG
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__=="__main__":
    dl = DataLoader()

    dataset = dl.load_data("/content/BrainTumorDataset/Mat_Format_Dataset/",
                           "mat", height=160, width=160)

    
    dataset = dl.configure_for_performance(dataset, shuffle=True)
    Xs = []
    ys = []
    labels = []
    for _x, _y, label in dataset:

        Xs.append(_x)
        ys.append(_y)
        labels.append(label)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    labels = np.concatenate(labels, axis=0)

    y = tf.keras.utils.to_categorical(y, 2)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    model = UNet((160, 160, 3), 2)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    
   
    #To test : Dice loss, BCE-Dice loss, Jaccard loss, Focal loss
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.AUC(), tf.keras.metrics.Recall()])


    epochs = 6
    model.fit(X_train, y_train, epochs=epochs, batch_size=8, validation_data=(X_test, y_test))


    def get_segmented_part(img, mask):

        zeros = tf.zeros_like(mask)
        boolean_mask = tf.cast(tf.math.not_equal(mask, zeros),tf.float32)

        return tf.math.multiply(img, boolean_mask)


    print('VGG Begining')

    ds = 
    mask_reshape = tf.map_fn(fn=lambda img: tf.image.resize(img, [224, 224]), y_train)

    vgg = VGG((224, 224, 3))
# print(X_test.shape)
# j = 1
# for i in range(0, 20, 2):
#     if ((i+1) == 11) or ((i+2)==1):
#         break
#     y_pred = model.predict(np.expand_dims(X_test[i], 0))

#     fig = plt.figure(figsize=(14, 14))
#     fig.add_subplot(5, 2, i+1)
#     plt.imshow(y_pred[0][:,:,0])
#     plt.axis('off')
#     plt.title("Pred")

#     fig.add_subplot(5, 2, i+2 )
#     plt.imshow(y_test[i][:,:,0])
#     plt.axis('off')
#     plt.title("gt")
#     j += 2

#     fig.save(f'figure{i}')

# plt.show()
