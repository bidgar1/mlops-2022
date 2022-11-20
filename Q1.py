from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics, tree
from sklearn.datasets import load_iris
import pytest

iris = load_iris()
X, y = iris.data, iris.target

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, random_state=42)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, random_state=0)


def validate(train1, train2):
    assert train1.all() == train2.all()


validate(X_train1,X_train2)
