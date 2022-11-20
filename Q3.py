
import argparse
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import datasets, svm, metrics, tree
from joblib import dump
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--clf_name')
parser.add_argument('--random_state')
args = parser.parse_args()


iris = load_iris()
X, y = iris.data, iris.target

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y, test_size=0.3, random_state = args.random_state, shuffle=False)

def train_model_svm(x_train, y_train, x_test, y_test):
    svm_model = svm.SVC()
    svm_model.fit(x_train, y_train)
    svm_predicted_train = svm_model.predict(x_test)
    train_metric = metrics.accuracy_score(y_pred=svm_predicted_train, y_true=y_test)
    return train_metric;

def train_model_tree(x_train, y_train, x_test, y_test):
    tree_model = tree.DecisionTreeClassifier()
    tree_model.fit(x_train, y_train)
    tree_predicted_train = tree_model.predict(x_test)
    train_metric = metrics.accuracy_score(y_pred=tree_predicted_train, y_true=y_test)
    np.append(tree_predicted_train,train_metric)
    return train_metric;   

if args.clf_name == 'svm':
    predicted_value_model = train_model_svm(X_train1,y_train1,X_test1,y_test1)
    print('test_accuracy: ',predicted_value_model)

if args.clf_name == 'tree':
    predicted_value_model = train_model_tree(X_train1,y_train1,X_test1,y_test1)
    print('test_accuracy: ',predicted_value_model)