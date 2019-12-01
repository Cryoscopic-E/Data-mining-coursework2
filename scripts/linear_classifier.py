import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report


class Output:
    def __init__(self, c_matrix, report, accuracy):
        self.confusion_matrix = c_matrix
        self.report = report
        self.accuracy = accuracy


def lin_classifier(_loss, _penalty, _max_iter):
    return linear_model.SGDClassifier(loss=_loss, max_iter=_max_iter, penalty=_penalty, tol=1e-3, n_jobs=-1)


def using_test_set(train_set, labels, test_set, labels_test, _clf):
    _clf.fit(train_set, labels)
    predictions = _clf.predict(test_set)
    cm = confusion_matrix(labels_test, predictions, labels=range(10))
    report = classification_report(labels_test, predictions)
    accuracy = _clf.score(test_set, labels_test)
    return Output(cm, report, accuracy)


def k_fold_validation(x, y, _clf, k=10):
    outputs = list()
    for train_index, test_index in KFold(k).split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        _clf.fit(x_train, y_train)
        predictions = _clf.predict(x_test)
        cm = confusion_matrix(y_test, predictions, labels=range(10))
        report = classification_report(y_test, predictions)
        accuracy = _clf.score(x_test, y_test)
        outputs.append(Output(cm, report, accuracy))
    return outputs


if __name__ == "__main__":

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

    losses = ['hinge', 'perceptron']
    penalties = ['l2', 'elasticnet']
    max_iterations = [500, 1000]

    for loss in losses:
        for penalty in penalties:
            for max_iter in max_iterations:
                clf = lin_classifier(loss, penalty, max_iter)
                # using 10 fold validation
                out_kf = k_fold_validation(X_full, y_full, clf)
                with open('../output/linear_classifier/10_fold_validation/output.txt', 'a') as file:
                    file.write(f'\nloss={loss}. penalty={penalty}, max_iteration={max_iter}\n')
                    for out, i in zip(out_kf, range(len(out_kf))):
                        file.write(f'\nFOLD {i}\n')
                        file.write(f'\nREPORT ----------------------\n{out.report}')
                        file.write(f'\nCONFUSION MATRIX ----------------------\n{out.confusion_matrix}')
                        file.write(f'\nACCURACY ----------------------\n{out.accuracy}')
                        file.write('\n================================================\n')

                # using test sets
                out = using_test_set(X_full, y_full, X_full_test, y_full_test, clf)
                with open('../output/linear_classifier/test_sets/output_full.txt', 'a') as file:
                    file.write(f'\nloss={loss}, penalty={penalty}, max_iteration={max_iter}\n')
                    file.write(f'\nREPORT ----------------------\n{out.report}')
                    file.write(f'\nCONFUSION MATRIX ----------------------\n{out.confusion_matrix}')
                    file.write(f'\nACCURACY ----------------------\n{out.accuracy}')
                    file.write('\n================================================\n')

                # using test set -4000
                out = using_test_set(X_4000, y_4000, X_4000_test, y_4000_test, clf)
                with open('../output/linear_classifier/test_sets/output_4000.txt', 'a') as file:
                    file.write(f'\nloss={loss}. penalty={penalty}, max_iteration={max_iter}\n')
                    file.write(f'\nREPORT ----------------------\n{out.report}')
                    file.write(f'\nCONFUSION MATRIX ----------------------\n{out.confusion_matrix}')
                    file.write(f'\nACCURACY ----------------------\n{out.accuracy}')
                    file.write('\n================================================\n')
                # using test set -9000
                out = using_test_set(X_9000, y_9000, X_9000_test, y_9000_test, clf)
                with open('../output/linear_classifier/test_sets/output_9000.txt', 'a') as file:
                    file.write(f'\nloss={loss}. penalty={penalty}, max_iteration={max_iter}\n')
                    file.write(f'\nREPORT ----------------------\n{out.report}')
                    file.write(f'\nCONFUSION MATRIX ----------------------\n{out.confusion_matrix}')
                    file.write(f'\nACCURACY ----------------------\n{out.accuracy}')
                    file.write('\n================================================\n')
