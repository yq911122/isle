import importHelper
cv = importHelper.load("cvMachine")

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn.ensemble

from numpy.random import RandomState
from sklearn.datasets import load_boston, load_diabetes, load_digits, load_iris
def test_randomForestRegressor():

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = RandomForestRegressor(n_estimators=40, alpha=20, max_features='auto')
    clf2 = sklearn.ensemble.RandomForestRegressor(n_estimators=40, max_features='auto')

    print cv.cross_validation(clf, X, y, scoring='r2')
    print cv.cross_validation(clf2, X, y, scoring='r2')

def test_AdaBoostRegressor():

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = AdaBoostRegressor(n_estimators=40, alpha=100)
    clf2 = sklearn.ensemble.AdaBoostRegressor(n_estimators=40)

    print cv.cross_validation(clf, X, y, scoring='r2')
    print cv.cross_validation(clf2, X, y, scoring='r2')

def test_GradientBoostingRegressor():

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    r = RandomState()
    clf = GradientBoostingRegressor(n_estimators=50, alpha=10, random_state=r)
    clf2 = sklearn.ensemble.GradientBoostingRegressor(n_estimators=50, random_state=r)
    
    print cv.group_cross_validaiton([clf, clf2], X, y, scoring='r2')
    # print cv.cross_validation(clf, X, y, scoring='r2')
    # print cv.cross_validation(clf2, X, y, scoring='r2')

def test_RandomForestClassifier():

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    r = RandomState()
    clf = RandomForestClassifier(n_estimators=10, alpha=0.001, random_state=r)
    clf2 = sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=r)

    # clf.fit(X_train, y_train)
    # print
    # clf2.fit(X_train, y_train)

    # print cv.group_cross_validaiton([clf, clf2], X, y, scoring='r2')
    print cv.cross_validation(clf, X, y, scoring='accuracy')
    print cv.cross_validation(clf2, X, y, scoring='accuracy')

def test_AdaBoostClassifier():

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    r = RandomState()
    clf = AdaBoostClassifier(n_estimators=25, alpha=0.001, random_state=r)
    clf2 = sklearn.ensemble.AdaBoostClassifier(n_estimators=25, random_state=r)

    # clf.fit(X_train, y_train)
    # print
    # clf2.fit(X_train, y_train)

    # print cv.group_cross_validaiton([clf, clf2], X, y, scoring='r2')
    print cv.cross_validation(clf, X, y, scoring='accuracy')
    print cv.cross_validation(clf2, X, y, scoring='accuracy')

def test_GradientBoostingClassifier():

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    r = RandomState()
    clf = GradientBoostingClassifier(n_estimators=15, alpha=0.001, random_state=r)
    clf2 = sklearn.ensemble.GradientBoostingClassifier(n_estimators=15, random_state=r)    
    
    # clf.fit(X_train, y_train)
    # # clf.predict_proba(X_test)
    # printy = np.array(original_y == k, dtype=np.float64)
    # clf2.fit(X_train, y_train)
    # clf2.staged_predict_proba(X_test)
    # print cv.group_cross_validaiton([clf, clf2], X, y, scoring='accuracy')
    print cv.cross_validation(clf, X, y, scoring='accuracy')
    print cv.cross_validation(clf2, X, y, scoring='accuracy')

