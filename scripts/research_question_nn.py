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
        '/Users/lusy/Desktop/Data-mining-coursework2/data/new_test/x_newtest_normalized.csv', header = None).values 
    test_labels = pd.read_csv('/Users/lusy/Desktop/Data-mining-coursework2/data/new_test/y_newtest_rnd.csv').values.ravel()
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

    output = ' '
    output += 'Min impurity decrease : {} \n'.format(min_impurity_decrease) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(confusion_matrix(y_test, y_pred)) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test, y_pred))
        accuracy_results.append(acc_score)
    with open(FOLDER_STRUCT + "/impurity/output_impurity.txt", "w") as text_file:
        text_file.write(output)

else:
    print('Invalid arguments.')
    print('Launch script with "-kfold" or "-testset"')





"""
if __name__ == '__main__':
    # add analysis from NN and Decision Trees

"""