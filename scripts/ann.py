import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold


class Output:
    def __init__(self, train_acc, train_loss, test_acc, test_loss):
        self.train_accuracy = train_acc
        self.train_loss = train_loss
        self.test_accuracy = test_acc
        self.test_loss = test_loss


def tf_model(_img_flat_size, n_neurons=128, _optimizer='adam'):
    _model = tf.keras.models.Sequential()
    _model.add(tf.keras.layers.Dense(_img_flat_size, input_shape=(_img_flat_size,), activation='relu'))
    _model.add(tf.keras.layers.Dense(n_neurons, activation='sigmoid'))
    _model.add(tf.keras.layers.Dense(10, activation='softmax'))

    _model.compile(loss='sparse_categorical_crossentropy', optimizer=_optimizer, metrics=['accuracy'])
    return _model


def k_fold_validation(x, y, _model, _epochs, k=10):
    outputs = list()
    for train_index, test_index in KFold(k).split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        history_train = _model.fit(x_train, y_train, epochs=_epochs)
        test_loss, test_accuracy = _model.evaluate(x_test, y_test, verbose=2)
        outputs.append(Output(history_train.history['acc'], history_train.history['loss'], test_accuracy, test_loss))
    return outputs


def use_test_set(x, y, x_test, y_test, _model, _epochs):
    history_train = _model.fit(x, y, epochs=_epochs)
    test_loss, test_accuracy = _model.evaluate(x_test, y_test, verbose=2)
    return Output(history_train.history['acc'], history_train.history['loss'], test_accuracy, test_loss)


if __name__ == '__main__':
    # full data
    X_full = pd.read_csv('../data/train/X_train_sliced_n_diced.csv').values / 255.0
    X_full_test = pd.read_csv('../data/test/X_test_sliced_n_diced.csv').values / 255.0
    y_full = pd.read_csv('../data/train/y_train_rnd.csv').values.ravel()
    y_full_test = pd.read_csv('../data/test/y_test_smpl.csv').values.ravel()
    # -4000
    X_4000 = pd.read_csv('../data/train/X_train_sliced_n_diced_less_4000.csv').values / 255.0
    X_4000_test = pd.read_csv('../data/test/X_test_sliced_n_diced_up_4000.csv').values / 255.0
    y_4000 = pd.read_csv('../data/train/y_train_rnd_less_4000.csv').values.ravel()
    y_4000_test = pd.read_csv('../data/test/y_test_rnd_up_4000.csv').values.ravel()
    # -9000
    X_9000 = pd.read_csv('../data/train/X_train_sliced_n_diced_less_9000.csv').values / 255.0
    X_9000_test = pd.read_csv('../data/test/X_test_sliced_n_diced_up_9000.csv').values / 255.0
    y_9000 = pd.read_csv('../data/train/y_train_rnd_less_9000.csv').values.ravel()
    y_9000_test = pd.read_csv('../data/test/y_test_rnd_up_9000.csv').values.ravel()

    epochs = [5, 10]
    number_neurons = [60, 120]
    optimizers = ['adam', 'sgd']
    img_size = len(X_full[0])

    # change optimizer
    for optimizer in optimizers:
        for epoch in epochs:
            model = tf_model(img_size, _optimizer=optimizer)
            # 10 fold
            out_kf = k_fold_validation(X_full, y_full, model, epoch)
            with open('../output/neural_network/10_fold_validation/output.txt', 'a') as file:
                file.write(f'\nEPOCHS={epoch}, OPTIMIZER={optimizer}\n')
                for out, i in zip(out_kf, range(len(out_kf))):
                    file.write(f'\nFOLD {i}\n')
                    file.write(f'\nTRAIN ACCURACY ------{out.train_accuracy}\n')
                    file.write(f'\nTRAIN LOSS ------{out.train_loss}\n')
                    file.write(f'\nTEST ACCURACY ------{out.test_accuracy}\n')
                    file.write(f'\nTEST LOSS ------{out.test_loss}\n')
                    file.write('\n================================================\n')

            # full
            out = use_test_set(X_full, y_full, X_full_test, y_full_test, model, epoch)
            with open('../output/neural_network/test_sets/output_full.txt', 'a') as file:
                file.write(f'\nEPOCHS={epoch}, OPTIMIZER={optimizer}\n')
                file.write(f'\nTRAIN ACCURACY ------{out.train_accuracy}\n')
                file.write(f'\nTRAIN LOSS ------{out.train_loss}\n')
                file.write(f'\nTEST ACCURACY ------{out.test_accuracy}\n')
                file.write(f'\nTEST LOSS ------{out.test_loss}\n')
                file.write('\n================================================\n')
            # -4000
            out = use_test_set(X_4000, y_4000, X_4000_test, y_4000_test, model, epoch)
            with open('../output/neural_network/test_sets/output_4000.txt', 'a') as file:
                file.write(f'\nEPOCHS={epoch}, OPTIMIZER={optimizer}\n')
                file.write(f'\nTRAIN ACCURACY ------{out.train_accuracy}\n')
                file.write(f'\nTRAIN LOSS ------{out.train_loss}\n')
                file.write(f'\nTEST ACCURACY ------{out.test_accuracy}\n')
                file.write(f'\nTEST LOSS ------{out.test_loss}\n')
                file.write('\n================================================\n')
            # -900
            out = use_test_set(X_9000, y_9000, X_9000_test, y_9000_test, model, epoch)
            with open('../output/neural_network/test_sets/output_9000.txt', 'a') as file:
                file.write(f'\nEPOCHS={epoch}, OPTIMIZER={optimizer}\n')
                file.write(f'\nTRAIN ACCURACY ------{out.train_accuracy}\n')
                file.write(f'\nTRAIN LOSS ------{out.train_loss}\n')
                file.write(f'\nTEST ACCURACY ------{out.test_accuracy}\n')
                file.write(f'\nTEST LOSS ------{out.test_loss}\n')
                file.write('\n================================================\n')

    # change number neurons
    for neurons in number_neurons:
        model = tf_model(img_size, n_neurons=neurons)
        for epoch in epochs:
            # 10 fold
            out_kf = k_fold_validation(X_full, y_full, model, epoch)
            with open('../output/neural_network/10_fold_validation/output_neurons.txt', 'a') as file:
                file.write(f'\nEPOCHS={epoch}, NEURONS={neurons}\n')
                for out, i in zip(out_kf, range(len(out_kf))):
                    file.write(f'\nFOLD {i}\n')
                    file.write(f'\nTRAIN ACCURACY ------{out.train_accuracy}\n')
                    file.write(f'\nTRAIN LOSS ------{out.train_loss}\n')
                    file.write(f'\nTEST ACCURACY ------{out.test_accuracy}\n')
                    file.write(f'\nTEST LOSS ------{out.test_loss}\n')
                    file.write('\n================================================\n')
            # full
            out = use_test_set(X_full, y_full, X_full_test, y_full_test, model, epoch)
            with open('../output/neural_network/test_sets/output_full_neurons.txt', 'a') as file:
                file.write(f'\nEPOCHS={epoch}, NEURONS={neurons}\n')
                file.write(f'\nTRAIN ACCURACY ------{out.train_accuracy}\n')
                file.write(f'\nTRAIN LOSS ------{out.train_loss}\n')
                file.write(f'\nTEST ACCURACY ------{out.test_accuracy}\n')
                file.write(f'\nTEST LOSS ------{out.test_loss}\n')
                file.write('\n================================================\n')
            # -4000
            out = use_test_set(X_4000, y_4000, X_4000_test, y_4000_test, model, epoch)
            with open('../output/neural_network/test_sets/output_4000_neurons.txt', 'a') as file:
                file.write(f'\nEPOCHS={epoch}, NEURONS={neurons}\n')
                file.write(f'\nTRAIN ACCURACY ------{out.train_accuracy}\n')
                file.write(f'\nTRAIN LOSS ------{out.train_loss}\n')
                file.write(f'\nTEST ACCURACY ------{out.test_accuracy}\n')
                file.write(f'\nTEST LOSS ------{out.test_loss}\n')
                file.write('\n================================================\n')
            # -900
            out = use_test_set(X_9000, y_9000, X_9000_test, y_9000_test, model, epoch)
            with open('../output/neural_network/test_sets/output_9000_neurons.txt', 'a') as file:
                file.write(f'\nEPOCHS={epoch}, NEURONS={neurons}\n')
                file.write(f'\nTRAIN ACCURACY ------{out.train_accuracy}\n')
                file.write(f'\nTRAIN LOSS ------{out.train_loss}\n')
                file.write(f'\nTEST ACCURACY ------{out.test_accuracy}\n')
                file.write(f'\nTEST LOSS ------{out.test_loss}\n')
                file.write('\n================================================\n')
