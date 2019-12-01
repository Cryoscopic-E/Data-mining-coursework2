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



X_train = pd.read_csv('data/train/X_train_sliced_n_diced.csv').values / 255.0
y_train = pd.read_csv('data/train/y_train_rnd.csv').values
X_test = pd.read_csv('data/new_test/x_test2_normalized.csv').values / 255.0
print(X_test.shape)
X1_test = pd.read_csv('data/test/X_test_sliced_n_diced.csv').values / 255.0
y_test = pd.read_csv('data/new_test/y_test2_rnd.csv').values
print(X1_test.shape)
print(y_test.shape)

FOLDER_STRUCT = 'data/test_train_2'

max_depths, accuracy_results = depth_tuning(X_train, y_train, X_test, y_test, 4, 25)
line_plot(max_depths, accuracy_results, 'Max Depth', FOLDER_STRUCT + '/max_depth/plot')

min_samples_splits, accuracy_results = samples_split_tuning(X_train, y_train, X_test, y_test)
line_plot(min_samples_splits, accuracy_results, 'Min Samples Split', FOLDER_STRUCT + '/sample_split/plot')

min_samples_leafs, accuracy_results = samples_leaf_tuning(X_train, y_train, X_test, y_test)
line_plot(min_samples_leafs, accuracy_results, 'Main Samples Leaf', FOLDER_STRUCT + '/sample_leaf/plot')

min_impurity_decreases, accuracy_results = min_impurity_tuning(X_train, y_train, X_test, y_test)
line_plot(min_impurity_decreases, accuracy_results, 'Min Impurity Decrease', FOLDER_STRUCT + '/impurity/plot')