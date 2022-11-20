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
    return svm_model, svm_predicted_train;

def train_model_tree(x_train, y_train, x_test, y_test):
    tree_model = tree.DecisionTreeClassifier()
    tree_model.fit(x_train, y_train)
    tree_predicted_train = tree_model.predict(x_test)
        
    return tree_model, tree_predicted_train;   

if args.clf_name == 'svm':
    model_svm, predicted_value_model = train_model_svm(X_train1,y_train1,X_test1,y_test1)
    print(predicted_value_model)
    dump(model_svm, 'model_svm')
    name_value = args.clf_name+'_'+args.random_state+'.txt'
    np.savetxt(name_value, predicted_value_model)

if args.clf_name == 'tree':
    model_tree, predicted_value_model = train_model_tree(X_train1,y_train1,X_test1,y_test1)
    print(predicted_value_model)
    dump(model_tree, 'model_tree')
    name_value = args.clf_name+'_'+args.random_state+'.txt'
    np.savetxt(name_value, predicted_value_model)


