import sys
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold


def tf_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        50, input_shape=(784,), activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


train = pd.read_csv(
    '/Users/lusy/Desktop/Data-mining-coursework2/data/train/X_train_sliced_n_diced.csv').values / 255.0
train_labels = pd.read_csv('/Users/lusy/Desktop/Data-mining-coursework2/data/train/y_train_rnd.csv').values.ravel()

if sys.argv[1] == '-testset':  # Using test set
    print('USING TEST SETS')
    test = pd.read_csv(
        '/Users/lusy/Desktop/Data-mining-coursework2/data/new_test/x_test2_normalized.csv').values / 255.0
    test_labels = pd.read_csv('/Users/lusy/Desktop/Data-mining-coursework2/data/new_test/y_test2_rnd.csv').values.ravel()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        2304, input_shape=(784,), activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(train, train_labels, epochs=5)
    print(test.shape)
    test_loss, test_accuracy = model.evaluate(test, test_labels, verbose=2)
    
    
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
else:
    print('Invalid arguments.')
    print('Launch script with "-kfold" or "-testset"')





"""
if __name__ == '__main__':
    # add analysis from NN and Decision Trees

"""