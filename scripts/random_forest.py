from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.legend_handler import HandlerLine2D
import seaborn as sn


def kfolds_random_forest(X, y, n_splits, n_estimators, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, out_file):
    """
    performs k fold cross validation on random forest
    :param X:
    :param y:
    :param n_splits:
    :param n_estimators:
    :param max_depth:
    :param min_samples_split:
    :param min_samples_leaf:
    :param min_impurity_decrease:
    :param out_file:
    :return:
    """
    kf = KFold(n_splits=n_splits)
    rnd_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,min_impurity_decrease=min_impurity_decrease)
    output = ""
    best_score = 0
    average_score = 0
    fold_num = 1
    for train_index, test_index in kf.split(X, y.ravel()):
        clone_clf = clone(rnd_clf)
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        acc_score = accuracy_score(y_test_fold, y_pred)
        if acc_score >= best_score:
            best_score = acc_score
        average_score += acc_score
        output += 'fold number {} scoring \n'.format(fold_num) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(confusion_matrix(y_test_fold, y_pred)) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test_fold, y_pred))
        fold_num += 1
    average_score /= n_splits
    output += 'av. accuracy score : {} \n best accuracy score : {} \n \n'.format(average_score, best_score)
    with open(out_file + ".txt", "w") as text_file:
        text_file.write(output)
    return average_score, best_score


def estiamtor_tuning_cv(X, y, kfold):
    """
    max. depth tuning for cross validation on random forest
    """
    num_estimators = np.arange(1, 30, 1)
    accuracy_results = []
    for estimators in num_estimators:
        average_score, best_score = kfolds_random_forest(X, y.ravel(), kfold, estimators, 100, 2, 1, 1e-7,
                                                         FOLDER_STRUCT + "/estimators/num_estimators_" + str(estimators))
        accuracy_results.append(average_score)
    return num_estimators, accuracy_results


def estiamtor_tuning(X_train, y_train, X_test, y_test):
    """
    max. depth tuning for random forest
    """
    num_estimators = np.arange(1, 30, 1)
    accuracy_results = []
    output = ''
    for estimators in num_estimators:
        rnd_clf = RandomForestClassifier(n_estimators=estimators, max_depth=100,
                                         min_samples_split=2,
                                         min_samples_leaf=1, min_impurity_decrease=1e-7)
        rnd_clf.fit(X_train, y_train.ravel())
        y_pred = rnd_clf.predict(X_test)
        acc_score = accuracy_score(y_test.ravel(), y_pred)
        cm = confusion_matrix(y_test, y_pred)
        outfile = FOLDER_STRUCT + "/estimators/num_estimators_" + str(estimators)
        confusion_matrix_heat_map(cm, outfile)
        output += 'Num Estimators : {} \n'.format(estimators) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(cm) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test, y_pred))
        accuracy_results.append(acc_score)
    with open(FOLDER_STRUCT + "/estimators/num_estimators.txt", "w") as text_file:
        text_file.write(output)
    return num_estimators, accuracy_results


def depth_tuning_cv(X, y, kfold, min_depth, max_depth):
    """
    max. depth tuning for cross validation on random forest
    """
    max_depths = np.arange(min_depth,max_depth,1)
    accuracy_results = []
    for max_depth in max_depths:
        average_score, best_score = kfolds_random_forest(X, y.ravel(), kfold, 10, max_depth, 2, 1, 1e-7,
                                                         FOLDER_STRUCT + "/max_depth/output_max_depth_" + str(max_depth))
        accuracy_results.append(average_score)
    return max_depths, accuracy_results


def depth_tuning(X_train, y_train, X_test, y_test, min_depth, max_depth):
    """
    max. depth tuning for random forest
    """
    max_depths = np.arange(min_depth, max_depth, 1)
    accuracy_results = []
    output = ''
    for max_depth in max_depths:
        rnd_clf = RandomForestClassifier(n_estimators=10, max_depth=max_depth,
                                         min_samples_split=2,
                                         min_samples_leaf=1, min_impurity_decrease=1e-7)
        rnd_clf.fit(X_train, y_train.ravel())
        y_pred = rnd_clf.predict(X_test)
        acc_score = accuracy_score(y_test.ravel(), y_pred)
        cm = confusion_matrix(y_test, y_pred)
        outfile = FOLDER_STRUCT + "/max_depth/output_max_depth_" + str(max_depth)
        confusion_matrix_heat_map(cm, outfile)
        output += 'Max depth : {} \n'.format(max_depth) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(cm) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test, y_pred))
        accuracy_results.append(acc_score)
    with open(FOLDER_STRUCT + "/max_depth/output_max_depth.txt", "w") as text_file:
        text_file.write(output)
    return max_depths, accuracy_results


def samples_split_tuning_cv(X, y, kfold):
    """
    sample split tuning for cross validation on random forest
    """
    min_samples_splits = np.linspace(0.01, 1.0, 20, endpoint=True)
    accuracy_results = []
    for min_samples_split in min_samples_splits:
        average_score, best_score = kfolds_random_forest(X, y.ravel(), kfold, 10, 100, min_samples_split, 1, 1e-7,
                                                         FOLDER_STRUCT + "/sample_split/output_sample_split_" + str(min_samples_split))
        accuracy_results.append(average_score)
    return min_samples_splits, accuracy_results


def samples_split_tuning(X_train, y_train, X_test, y_test):
    """
    min samples split tuning for random forest
    """
    min_samples_splits = np.linspace(0.01, 1.0, 20, endpoint=True)
    accuracy_results = []
    output = ''
    for min_samples_split in min_samples_splits:
        rnd_clf = RandomForestClassifier(n_estimators=10, max_depth=100,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=1, min_impurity_decrease=1e-7)
        rnd_clf.fit(X_train, y_train.ravel())
        y_pred = rnd_clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        outfile = FOLDER_STRUCT + "/sample_split/output_sample_split_" + str(min_samples_split)
        confusion_matrix_heat_map(cm, outfile)
        output += 'Min sample split : {} \n'.format(min_samples_split) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(cm) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test, y_pred))
        accuracy_results.append(acc_score)
    with open(FOLDER_STRUCT + "/sample_split/output_sample_split.txt", "w") as text_file:
        text_file.write(output)
    return min_samples_splits, accuracy_results


def samples_leaf_tuning_cv(X, y, kfold):
    """
    min samples leaf tuning for cross validation on random forest
    """
    min_samples_leafs = np.linspace(0.01, 0.1, 5, endpoint=True)
    accuracy_results = []
    for min_samples_leaf in min_samples_leafs:
        average_score, best_score = kfolds_random_forest(X, y.ravel(), kfold, 10, 100, 2, min_samples_leaf, 1e-7,
                                                         FOLDER_STRUCT + "/sample_leaf/output_sample_leaf_" + str(min_samples_leaf))
        accuracy_results.append(average_score)
    return min_samples_leafs, accuracy_results


def samples_leaf_tuning(X_train, y_train, X_test, y_test):
    """
    min samples leaf tuning for random forest
    """
    min_samples_leafs = np.linspace(0.01, 0.1, 5, endpoint=True)
    accuracy_results = []
    output = ''
    for min_samples_leaf in min_samples_leafs:
        rnd_clf = RandomForestClassifier(n_estimators=10, max_depth=100,
                                         min_samples_split=2,
                                         min_samples_leaf=min_samples_leaf, min_impurity_decrease=1e-7)
        rnd_clf.fit(X_train, y_train.ravel())
        y_pred = rnd_clf.predict(X_test)
        acc_score = accuracy_score(y_test.ravel(), y_pred)
        cm = confusion_matrix(y_test, y_pred)
        outfile = FOLDER_STRUCT + "/sample_leaf/output_sample_leaf_" + str(min_samples_leaf)
        confusion_matrix_heat_map(cm, outfile)
        output += 'min samples leaf : {} \n'.format(min_samples_leaf) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(cm) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test, y_pred))
        accuracy_results.append(acc_score)
    with open(FOLDER_STRUCT + "/sample_leaf/output_sample_leaf.txt", "w") as text_file:
        text_file.write(output)
    return min_samples_leafs, accuracy_results


def min_impurity_tuning_cv(X, y, kfold):
    """
    min simpurity tuning (effectively pruning) for cross validation on random forest
    """
    min_impurity_decreases = np.linspace(0, 10, 11, endpoint=True)
    accuracy_results = []
    for min_impurity_decrease in min_impurity_decreases:
        average_score, best_score = kfolds_random_forest(X, y.ravel(), kfold, 10, 100, 2, 1, min_impurity_decrease,
                                                         FOLDER_STRUCT + "/impurity/output_impurity_" + str(min_impurity_decrease))
        accuracy_results.append(average_score)
    return min_impurity_decreases, accuracy_results


def min_impurity_tuning(X_train, y_train, X_test, y_test):
    """
    min simpurity tuning (effectively pruning) for random forest
    """
    min_impurity_decreases = np.linspace(0, 10, 11, endpoint=True)
    accuracy_results = []
    output = ''
    for min_impurity_decrease in min_impurity_decreases:
        rnd_clf = RandomForestClassifier(n_estimators=10, max_depth=100,
                                         min_samples_split=2,
                                         min_samples_leaf=1, min_impurity_decrease=min_impurity_decrease)
        rnd_clf.fit(X_train, y_train.ravel())
        y_pred = rnd_clf.predict(X_test)
        acc_score = accuracy_score(y_test.ravel(), y_pred)
        cm = confusion_matrix(y_test, y_pred)
        outfile = FOLDER_STRUCT + "/impurity/output_impurity_" + str(min_impurity_decrease)
        confusion_matrix_heat_map(cm, outfile)
        output += 'Min impurity decrease : {} \n'.format(min_impurity_decrease) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(cm) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test, y_pred))
        accuracy_results.append(acc_score)
    with open(FOLDER_STRUCT + "/impurity/output_impurity.txt", "w") as text_file:
        text_file.write(output)
    return min_impurity_decreases, accuracy_results


def line_plot(x, y, xlabel, out_file):
    """
    plot and save line plot
    """
    fig = plt.figure()
    plt.plot(x, y)
    plt.ylabel('Accuracy Score')
    plt.xlabel(xlabel)
    fig.savefig(out_file + '.png')
    plt.close(fig)


def two_line_plot(x, y1, y2, xlabel, out_file):
    """
    plot and save line plot
    """
    fig = plt.figure()
    line1, = plt.plot(x, y1, 'b', label='Train accuracy')
    line2, = plt.plot(x, y2, 'r', label='Test accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy Score')
    plt.xlabel(xlabel)
    fig.savefig(out_file + '.png')
    plt.close(fig)

def confusion_matrix_heat_map(cm, outfile):
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    fig.savefig(outfile + '_cm.png')
    plt.close(fig)

# All testing below:

# 10 fold cross validation experiments for baseline

FOLDER_STRUCT = 'sherwood/cross_validation'

X_train = pd.read_csv('data/train/X_train_sliced_n_diced.csv').values / 255.0
y_train = pd.read_csv('data/train/y_train_rnd.csv').values
X_test = pd.read_csv('data/test/X_test_sliced_n_diced.csv').values / 255.0
y_test = pd.read_csv('data/test/y_test_rnd.csv').values

num_estimators, num_estimators_accuracy_results_cv = estiamtor_tuning_cv(X_train, y_train, 10)
line_plot(num_estimators, num_estimators_accuracy_results_cv, 'Forest Size', FOLDER_STRUCT + '/estimators/plot')

max_depths, max_depths_accuracy_results_cv = depth_tuning_cv(X_train, y_train, 10, 4, 25)
line_plot(max_depths, max_depths_accuracy_results_cv, 'Max Depth', FOLDER_STRUCT + '/max_depth/plot')

min_samples_splits, min_samples_splits_accuracy_results_cv = samples_split_tuning_cv(X_train, y_train, 10)
line_plot(min_samples_splits, min_samples_splits_accuracy_results_cv, 'Min Samples Split', FOLDER_STRUCT + '/sample_split/plot')

min_samples_leafs, min_samples_leafs_accuracy_results_cv = samples_leaf_tuning_cv(X_train, y_train, 10)
line_plot(min_samples_leafs, min_samples_leafs_accuracy_results_cv, 'Min Samples Leaf', FOLDER_STRUCT + '/sample_leaf/plot')

min_impurity_decreases, min_impurity_decreases_accuracy_results_cv = min_impurity_tuning_cv(X_train, y_train, 10)
line_plot(min_impurity_decreases, min_impurity_decreases_accuracy_results_cv, 'Min Impurity Decrease', FOLDER_STRUCT + '/impurity/plot')

# using training set and test as given
FOLDER_STRUCT = 'sherwood/test_train_1'

num_estimators, num_estimators_accuracy_results = estiamtor_tuning(X_train, y_train, X_test, y_test)
two_line_plot(num_estimators, num_estimators_accuracy_results_cv, num_estimators_accuracy_results, 'Forest Size', FOLDER_STRUCT + '/estimators/plot')
#
max_depths, max_depths_accuracy_results = depth_tuning(X_train, y_train, X_test, y_test, 4, 25)
two_line_plot(max_depths, max_depths_accuracy_results_cv, max_depths_accuracy_results, 'Max Depth', FOLDER_STRUCT + '/max_depth/plot')

min_samples_splits, min_samples_splits_accuracy_results = samples_split_tuning(X_train, y_train, X_test, y_test)
two_line_plot(min_samples_splits, min_samples_splits_accuracy_results_cv, min_samples_splits_accuracy_results, 'Min Samples Split', FOLDER_STRUCT + '/sample_split/plot')

min_samples_leafs, min_samples_leafs_accuracy_results = samples_leaf_tuning(X_train, y_train, X_test, y_test)
two_line_plot(min_samples_leafs, min_samples_leafs_accuracy_results_cv, min_samples_leafs_accuracy_results, 'Main Samples Leaf', FOLDER_STRUCT + '/sample_leaf/plot')

min_impurity_decreases, min_impurity_decreases_accuracy_results = min_impurity_tuning(X_train, y_train, X_test, y_test)
two_line_plot(min_impurity_decreases, min_impurity_decreases_accuracy_results_cv, min_impurity_decreases_accuracy_results, 'Min Impurity Decrease', FOLDER_STRUCT + '/impurity/plot')

# reduced training set and increased test set
FOLDER_STRUCT = 'sherwood/test_train_2'

X_train = pd.read_csv('data/train/X_train_sliced_n_diced_less_4000.csv').values / 255.0
y_train = pd.read_csv('data/train/y_train_rnd_less_4000.csv').values
X_test = pd.read_csv('data/test/X_test_sliced_n_diced_up_4000.csv').values / 255.0
y_test = pd.read_csv('data/test/y_test_rnd_up_4000.csv').values

num_estimators, num_estimators_accuracy_results_cv = estiamtor_tuning_cv(X_train, y_train, 10)
num_estimators, num_estimators_accuracy_results = estiamtor_tuning(X_train, y_train, X_test, y_test)
two_line_plot(num_estimators, num_estimators_accuracy_results_cv, num_estimators_accuracy_results, 'Forest Size', FOLDER_STRUCT + '/estimators/plot')

max_depths, max_depths_accuracy_results_cv = depth_tuning_cv(X_train, y_train, 10, 4, 25)
max_depths, max_depths_accuracy_results = depth_tuning(X_train, y_train, X_test, y_test, 4, 25)
two_line_plot(max_depths, max_depths_accuracy_results_cv, max_depths_accuracy_results, 'Max Depth', FOLDER_STRUCT + '/max_depth/plot')

min_samples_splits, min_samples_splits_accuracy_results_cv = samples_split_tuning_cv(X_train, y_train, 10)
min_samples_splits, min_samples_splits_accuracy_results = samples_split_tuning(X_train, y_train, X_test, y_test)
two_line_plot(min_samples_splits, min_samples_splits_accuracy_results_cv, min_samples_splits_accuracy_results, 'Min Samples Split', FOLDER_STRUCT + '/sample_split/plot')

min_samples_leafs, min_samples_leafs_accuracy_results_cv = samples_leaf_tuning_cv(X_train, y_train, 10)
min_samples_leafs, min_samples_leafs_accuracy_results = samples_leaf_tuning(X_train, y_train, X_test, y_test)
two_line_plot(min_samples_leafs, min_samples_leafs_accuracy_results_cv, min_samples_leafs_accuracy_results, 'Main Samples Leaf', FOLDER_STRUCT + '/sample_leaf/plot')

min_impurity_decreases, min_impurity_decreases_accuracy_results_cv = min_impurity_tuning_cv(X_train, y_train, 10)
min_impurity_decreases, min_impurity_decreases_accuracy_results = min_impurity_tuning(X_train, y_train, X_test, y_test)
two_line_plot(min_impurity_decreases, min_impurity_decreases_accuracy_results_cv, min_impurity_decreases_accuracy_results, 'Min Impurity Decrease', FOLDER_STRUCT + '/impurity/plot')

# reduced training set and increased test set
FOLDER_STRUCT = 'sherwood/test_train_3'

X_train = pd.read_csv('data/train/X_train_sliced_n_diced_less_9000.csv').values / 255.0
y_train = pd.read_csv('data/train/y_train_rnd_less_9000.csv').values
X_test = pd.read_csv('data/test/X_test_sliced_n_diced_up_9000.csv').values / 255.0
y_test = pd.read_csv('data/test/y_test_rnd_up_9000.csv').values

num_estimators, num_estimators_accuracy_results_cv = estiamtor_tuning_cv(X_train, y_train, 10)
num_estimators, num_estimators_accuracy_results = estiamtor_tuning(X_train, y_train, X_test, y_test)
two_line_plot(num_estimators, num_estimators_accuracy_results_cv, num_estimators_accuracy_results, 'Forest Size', FOLDER_STRUCT + '/estimators/plot')

max_depths, max_depths_accuracy_results_cv = depth_tuning_cv(X_train, y_train, 10, 4, 25)
max_depths, max_depths_accuracy_results = depth_tuning(X_train, y_train, X_test, y_test, 4, 25)
two_line_plot(max_depths, max_depths_accuracy_results_cv, max_depths_accuracy_results, 'Max Depth', FOLDER_STRUCT + '/max_depth/plot')

min_samples_splits, min_samples_splits_accuracy_results_cv = samples_split_tuning_cv(X_train, y_train, 10)
min_samples_splits, min_samples_splits_accuracy_results = samples_split_tuning(X_train, y_train, X_test, y_test)
two_line_plot(min_samples_splits, min_samples_splits_accuracy_results_cv, min_samples_splits_accuracy_results, 'Min Samples Split', FOLDER_STRUCT + '/sample_split/plot')

min_samples_leafs, min_samples_leafs_accuracy_results_cv = samples_leaf_tuning_cv(X_train, y_train, 10)
min_samples_leafs, min_samples_leafs_accuracy_results = samples_leaf_tuning(X_train, y_train, X_test, y_test)
two_line_plot(min_samples_leafs, min_samples_leafs_accuracy_results_cv, min_samples_leafs_accuracy_results, 'Main Samples Leaf', FOLDER_STRUCT + '/sample_leaf/plot')

min_impurity_decreases, min_impurity_decreases_accuracy_results_cv = min_impurity_tuning_cv(X_train, y_train, 10)
min_impurity_decreases, min_impurity_decreases_accuracy_results = min_impurity_tuning(X_train, y_train, X_test, y_test)
two_line_plot(min_impurity_decreases, min_impurity_decreases_accuracy_results_cv, min_impurity_decreases_accuracy_results, 'Min Impurity Decrease', FOLDER_STRUCT + '/impurity/plot')
