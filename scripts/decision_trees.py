import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF
import graphviz


def prune_index(inner_tree, index, threshold):
    """
    prunes tree for given threshold value (type depends on criterion: either gini or entropy)
    :param inner_tree: tree to prune
    :param index: starting index
    :param threshold: value to prune for
    :return: none, tree operated on has been mutated
    """
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are children, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)


def kfolds_decision_tree(X, y, n_splits, criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, out_file):
    """
    performs kfolds cross validation on decision tree
    :param X: to classify
    :param y: labels
    :param n_splits: numbe rof folds for cv
    :param criterion: gini or entropy
    :param max_depth: max depth of tree
    :param min_samples_split: min number of samples needed at split
    :param min_samples_leaf: min number of samples at leaf node
    :param min_impurity_decrease: min impurity decrease
    :param out_file: filepath
    :return: mean score and best score
    """
    kf = KFold(n_splits=n_splits)
    dt = tree.DecisionTreeClassifier(criterion=criterion,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     min_impurity_decrease=min_impurity_decrease)
    output = ""
    best_score = 0
    average_score = 0
    fold_num = 1
    for train_index, test_index in kf.split(X, y):
        clone_clf = clone(dt)
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


def depth_tuning_cv(X, y, kfold, min_depth, max_depth):
    """
    max. depth tuning for cross validation on decision trees
    """
    max_depths = np.arange(min_depth,max_depth,1)
    accuracy_results = []
    for max_depth in max_depths:
        average_score, best_score = kfolds_decision_tree(X, y, kfold, 'gini', max_depth, 2, 1, 1e-7,
                                                         FOLDER_STRUCT + "/max_depth/output_max_depth_" + str(max_depth))
        accuracy_results.append(average_score)
    return max_depths, accuracy_results


def depth_tuning(X_train, y_train, X_test, y_test, min_depth, max_depth):
    """
    max. depth tuning for decision tree
    """
    max_depths = np.arange(min_depth, max_depth, 1)
    accuracy_results = []
    output = ''
    for max_depth in max_depths:
        clf = tree.DecisionTreeClassifier(criterion='gini',
                                         max_depth=max_depth,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         min_impurity_decrease=1e-7)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        output += 'Max depth : {} \n'.format(max_depth) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(confusion_matrix(y_test, y_pred)) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test, y_pred))
        accuracy_results.append(acc_score)
    with open(FOLDER_STRUCT + "/max_depth/output_max_depth.txt", "w") as text_file:
        text_file.write(output)
    return max_depths, accuracy_results


def samples_split_tuning_cv(X, y, kfold):
    """
    sample split tuning for cross validation on decision trees
    """
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    accuracy_results = []
    for min_samples_split in min_samples_splits:
        average_score, best_score = kfolds_decision_tree(X, y, kfold, 'gini', 100, min_samples_split, 1, 1e-7,
                                                         FOLDER_STRUCT + "/sample_split/output_sample_split_" + str(min_samples_split))
        accuracy_results.append(average_score)
    return min_samples_splits, accuracy_results


def samples_split_tuning(X_train, y_train, X_test, y_test):
    """
    min samples split tuning for decision trees
    """
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    accuracy_results = []
    output = ''
    for min_samples_split in min_samples_splits:
        clf = tree.DecisionTreeClassifier(criterion='gini',
                                      max_depth=100,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=1,
                                      min_impurity_decrease=1e-7)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        output += 'Min sample split : {} \n'.format(min_samples_split) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(confusion_matrix(y_test, y_pred)) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test, y_pred))
        accuracy_results.append(acc_score)
    with open(FOLDER_STRUCT + "/sample_split/output_sample_split.txt", "w") as text_file:
        text_file.write(output)
    return min_samples_splits, accuracy_results


def samples_leaf_tuning_cv(X, y, kfold):
    """
    min samples leaf tuning for cross validation on decision trees
    """
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    accuracy_results = []
    for min_samples_leaf in min_samples_leafs:
        average_score, best_score = kfolds_decision_tree(X, y, kfold, 'gini', 100, 2, min_samples_leaf, 1e-7,
                                                         FOLDER_STRUCT + "/sample_leaf/output_sample_leaf_" + str(min_samples_leaf))
        accuracy_results.append(average_score)
    return min_samples_leafs, accuracy_results


def samples_leaf_tuning(X_train, y_train, X_test, y_test):
    """
    min samples leaf tuning for  decision trees
    """
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    accuracy_results = []
    output = ''
    for min_samples_leaf in min_samples_leafs:
        clf = tree.DecisionTreeClassifier(criterion='gini',
                                      max_depth=100,
                                      min_samples_split=2,
                                      min_samples_leaf=min_samples_leaf,
                                      min_impurity_decrease=1e-7)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        output += 'min samples leaf : {} \n'.format(min_samples_leaf) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(confusion_matrix(y_test, y_pred)) \
                  + 'classification report : \n {} \n \n'.format(classification_report(y_test, y_pred))
        accuracy_results.append(acc_score)
    with open(FOLDER_STRUCT + "/sample_leaf/output_sample_leaf.txt", "w") as text_file:
        text_file.write(output)
    return min_samples_leafs, accuracy_results


def min_impurity_tuning_cv(X, y, kfold):
    """
    min simpurity tuning (effectively pruning) for cross validation on  decision trees
    """
    min_impurity_decreases = np.linspace(0, 10, 11, endpoint=True)
    accuracy_results = []
    for min_impurity_decrease in min_impurity_decreases:
        average_score, best_score = kfolds_decision_tree(X, y, kfold, 'gini', 100, 2, 1, min_impurity_decrease,
                                                         FOLDER_STRUCT + "/impurity/output_impurity_" + str(min_impurity_decrease))
        accuracy_results.append(average_score)
    return min_impurity_decreases, accuracy_results


def min_impurity_tuning(X_train, y_train, X_test, y_test):
    """
    min simpurity tuning (effectively pruning) for  decision trees
    """
    min_impurity_decreases = np.linspace(0, 10, 11, endpoint=True)
    accuracy_results = []
    output = ''
    for min_impurity_decrease in min_impurity_decreases:
        clf = tree.DecisionTreeClassifier(criterion='gini',
                                      max_depth=100,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_impurity_decrease=min_impurity_decrease)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        output += 'Min impurity decrease : {} \n'.format(min_impurity_decrease) \
                  + 'accuracy score : {} \n'.format(acc_score) \
                  + 'confusion matrix : \n {} \n'.format(confusion_matrix(y_test, y_pred)) \
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


# # visualisation - not decided on best way to show this yet
# dot_data = tree.export_graphviz(dt, out_file=None, filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render(filename=graph)


# All testing below:
FOLDER_STRUCT = 'cross_validation'

X_train = pd.read_csv('data/train/X_train_sliced_n_diced.csv').values / 255.0
y_train = pd.read_csv('data/train/y_train_rnd.csv').values
X_test = pd.read_csv('data/test/X_test_sliced_n_diced.csv').values / 255.0
y_test = pd.read_csv('data/test/y_test_rnd.csv').values

max_depths, accuracy_results = depth_tuning_cv(X_train, y_train, 10, 4, 25)
line_plot(max_depths, accuracy_results, 'Max Depth', FOLDER_STRUCT + '/max_depth/plot')

min_samples_splits, accuracy_results = samples_split_tuning_cv(X_train, y_train, 10)
line_plot(min_samples_splits, accuracy_results, 'Min Samples Split', FOLDER_STRUCT + '/sample_split/plot')

min_samples_leafs, accuracy_results = samples_leaf_tuning_cv(X_train, y_train, 10)
line_plot(min_samples_leafs, accuracy_results, 'Main Samples Leaf', FOLDER_STRUCT + '/sample_leaf/plot')

min_impurity_decreases, accuracy_results = min_impurity_tuning_cv(X_train, y_train, 10)
line_plot(min_impurity_decreases, accuracy_results, 'Min Impurity Decrease', FOLDER_STRUCT + '/impurity/plot')

FOLDER_STRUCT = 'test_train_1'

max_depths, accuracy_results = depth_tuning(X_train, y_train, X_test, y_test, 4, 25)
line_plot(max_depths, accuracy_results, 'Max Depth', FOLDER_STRUCT + '/max_depth/plot')

min_samples_splits, accuracy_results = samples_split_tuning(X_train, y_train, X_test, y_test)
line_plot(min_samples_splits, accuracy_results, 'Min Samples Split', FOLDER_STRUCT + '/sample_split/plot')

min_samples_leafs, accuracy_results = samples_leaf_tuning(X_train, y_train, X_test, y_test)
line_plot(min_samples_leafs, accuracy_results, 'Main Samples Leaf', FOLDER_STRUCT + '/sample_leaf/plot')

min_impurity_decreases, accuracy_results = min_impurity_tuning(X_train, y_train, X_test, y_test)
line_plot(min_impurity_decreases, accuracy_results, 'Min Impurity Decrease', FOLDER_STRUCT + '/impurity/plot')

FOLDER_STRUCT = 'test_train_2'

X_train = pd.read_csv('data/train/X_train_sliced_n_diced_less_4000.csv').values / 255.0
y_train = pd.read_csv('data/train/y_train_rnd_less_4000.csv').values
X_test = pd.read_csv('data/test/X_test_sliced_n_diced_up_4000.csv').values / 255.0
y_test = pd.read_csv('data/test/y_test_rnd_up_4000.csv').values

max_depths, accuracy_results = depth_tuning(X_train, y_train, X_test, y_test, 4, 25)
line_plot(max_depths, accuracy_results, 'Max Depth', FOLDER_STRUCT + '/max_depth/plot')

min_samples_splits, accuracy_results = samples_split_tuning(X_train, y_train, X_test, y_test)
line_plot(min_samples_splits, accuracy_results, 'Min Samples Split', FOLDER_STRUCT + '/sample_split/plot')

min_samples_leafs, accuracy_results = samples_leaf_tuning(X_train, y_train, X_test, y_test)
line_plot(min_samples_leafs, accuracy_results, 'Main Samples Leaf', FOLDER_STRUCT + '/sample_leaf/plot')

min_impurity_decreases, accuracy_results = min_impurity_tuning(X_train, y_train, X_test, y_test)
line_plot(min_impurity_decreases, accuracy_results, 'Min Impurity Decrease', FOLDER_STRUCT + '/impurity/plot')

FOLDER_STRUCT = 'test_train_3'

X_train = pd.read_csv('data/train/X_train_sliced_n_diced_less_9000.csv').values / 255.0
y_train = pd.read_csv('data/train/y_train_rnd_less_9000.csv').values
X_test = pd.read_csv('data/test/X_test_sliced_n_diced_up_9000.csv').values / 255.0
y_test = pd.read_csv('data/test/y_test_rnd_up_9000.csv').values

max_depths, accuracy_results = depth_tuning(X_train, y_train, X_test, y_test, 4, 25)
line_plot(max_depths, accuracy_results, 'Max Depth', FOLDER_STRUCT + '/max_depth/plot')

min_samples_splits, accuracy_results = samples_split_tuning(X_train, y_train, X_test, y_test)
line_plot(min_samples_splits, accuracy_results, 'Min Samples Split', FOLDER_STRUCT + '/sample_split/plot')

min_samples_leafs, accuracy_results = samples_leaf_tuning(X_train, y_train, X_test, y_test)
line_plot(min_samples_leafs, accuracy_results, 'Main Samples Leaf', FOLDER_STRUCT + '/sample_leaf/plot')

min_impurity_decreases, accuracy_results = min_impurity_tuning(X_train, y_train, X_test, y_test)
line_plot(min_impurity_decreases, accuracy_results, 'Min Impurity Decrease', FOLDER_STRUCT + '/impurity/plot')

# Run GridSearch to get best combination of results and then do some quick visualisation