import sys
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold


def tf_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        50, input_shape=(2304,), activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


train = pd.read_csv(
    '../data/train/x_train_gr_smpl_normalized.csv').values / 255.0
train_labels = pd.read_csv('../data/train/y_train_smpl.csv').values.ravel()

if sys.argv[1] == '-kfold':  # 10 fold validation
    print('USING 10 FOLD CROSSVALIDATION')
    for train_index, test_index in KFold(10).split(train):
        x_train, x_test = train[train_index], train[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]

        model = tf_model()
        model.fit(x_train, y_train, epochs=5)

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_accuracy)
elif sys.argv[1] == '-testset':  # Using test set
    print('USING TEST SETS')
    test = pd.read_csv(
        '../data/test/x_test_gr_smpl_normalized.csv').values / 255.0
    test_labels = pd.read_csv('../data/test/y_test_smpl.csv').values.ravel()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        2304, input_shape=(2304,), activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(train, train_labels, epochs=5)
    test_loss, test_accuracy = model.evaluate(test, test_labels, verbose=2)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
else:
    print('Invalid arguments.')
    print('Launch script with "-kfold" or "-testset"')