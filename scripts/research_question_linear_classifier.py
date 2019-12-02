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


if __name__ == "__main__":

    # full data
    X_full = pd.read_csv('data/train/X_train_sliced_n_diced.csv').values / 255.0
    X_full_test = pd.read_csv('data/new_test/x_newtest_normalized.csv', header = None).values 
    y_full = pd.read_csv('data/train/y_train_rnd.csv').values.ravel()
    y_full_test = pd.read_csv('data/new_test/y_newtest_rnd.csv').values.ravel()
  
    losses = ['hinge', 'perceptron']
    penalties = ['l2', 'elasticnet']
    max_iterations = [500, 1000]

    for loss in losses:
        for penalty in penalties:
            for max_iter in max_iterations:
                clf = lin_classifier(loss, penalty, max_iter)
               
                # using test sets
                out = using_test_set(X_full, y_full, X_full_test, y_full_test, clf)
                with open('output/research_linear_classifier/new_test.txt', 'a') as file:
                    file.write(f'\nloss={loss}, penalty={penalty}, max_iteration={max_iter}\n')
                    file.write(f'\nREPORT ----------------------\n{out.report}')
                    file.write(f'\nCONFUSION MATRIX ----------------------\n{out.confusion_matrix}')
                    file.write(f'\nACCURACY ----------------------\n{out.accuracy}')
                    file.write('\n================================================\n')

                
