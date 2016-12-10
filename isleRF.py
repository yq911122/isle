import warnings
from warnings import warn

import numpy as np

# from scipy.sparse import issparse
from sklearn.ensemble._gradient_boosting import predict_stages

from sklearn.ensemble.gradient_boosting import LeastSquaresError, BinomialDeviance, MultinomialDeviance, ExponentialLoss

from sklearn.utils import check_array, check_X_y, check_random_state, compute_sample_weight
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted

from islebase import PertubationSampler, ISLEBaseEnsembleRegressor, ISLEBaseEnsembleClassifier
# from loss import LeastSquaresError
from sklearn.tree._tree import DTYPE, DOUBLE
LOSS_FUNCTIONS = {'ls': LeastSquaresError,
                  'deviance': None,
                  'exponential': ExponentialLoss
                  }


class RandomForestRegressor(ISLEBaseEnsembleRegressor):
    """docstring for RandomForestRegressor"""
    def __init__(self, 
                 alpha=0.1,
                 n_estimators=10,  
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 random_state=None,
                 warm_start=False,
                 post_model=Lasso(),
                 learning_rate=1.0):
        super(RandomForestRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            alpha=alpha,
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes", "min_impurity_split",
                              "random_state"),
            random_state=random_state,
            bootstrap = bootstrap,
            warm_start=warm_start,
            post_model=post_model,
            learning_rate=learning_rate)
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split

        self.features_idx = []


    def _build_base_estimators(self, estimator, X, y):
        sampler = self.pertubater
        X_sample, y_sample, idx = sampler.sample_X_y(X, y, return_index=True)
        X_sample, feat_idx = sampler.sample_features(X_sample, return_index=True)
        self.features_idx.append(feat_idx)
        estimator.fit(X_sample, y_sample, sample_weight=self.sample_weight[idx])
        return estimator

    def _tree_predict(self, estimator_id, X):
        return self.estimators_[estimator_id].predict(X[:, self.features_idx[estimator_id]])       

    def init_stage(self):
        self.estimators_ = []
        self.features_idx = []

class RandomForestClassifier(ISLEBaseEnsembleClassifier):
    """docstring for RandomForestClassifier"""
    def __init__(self, 
                 alpha=0.1,
                 n_estimators=10,  
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 random_state=None,
                 warm_start=False,
                 post_model=Lasso(),
                 learning_rate=1.0):
        super(RandomForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            alpha=alpha,
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes", "min_impurity_split",
                              "random_state"),
            random_state=random_state,
            bootstrap = bootstrap,
            warm_start=warm_start,
            post_model=post_model,
            learning_rate=learning_rate)
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split

        self.features_idx = []

    def _build_base_estimators(self, estimator, X, y):
        sampler = self.pertubater
        X_sample, y_sample, idx = sampler.sample_X_y(X, y, return_index=True)
        X_sample, feat_idx = sampler.sample_features(X_sample, mode=None, return_index=True)
        self.features_idx.append(feat_idx)
        # print X_sample[:5]
        # print y_sample[:5]
        estimator.fit(X_sample, y_sample, sample_weight=self.sample_weight[idx])
        # print estimator.feature_importances_
        return estimator

    def init_stage(self):
        self.estimators_ = []
        self.features_idx = []

    def _tree_predict_proba(self, estimator_id, X):
        return self.estimators_[estimator_id].predict_proba(X[:, self.features_idx[estimator_id]])
        
class AdaBoostRegressor(ISLEBaseEnsembleRegressor):
    """docstring for AdaBoostRegressor"""
    def __init__(self, 
                 base_estimator=DecisionTreeRegressor(max_depth=3),
                 alpha=0.1,
                 n_estimators=10,
                 loss='linear',
                 random_state=None,
                 post_model=Lasso(),
                 learning_rate=1.0):
        super(AdaBoostRegressor, self).__init__(
                 base_estimator=base_estimator,
                 alpha=alpha,
                 n_estimators=n_estimators,
                 estimator_params=tuple(),
                 random_state=random_state,
                 bootstrap = True,
                 warm_start=False,
                 post_model=post_model,
                 learning_rate=learning_rate)
        self.loss = loss


    def _build_base_estimators(self, estimator, X, y):
        bootstrap_idx = self.pertubater.sample_array(X, self.random_state, self.sample_weight)
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (self.sample_weight * error_vect).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            self.sample_weight /= np.sum(self.sample_weight)
            return estimator

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
                if len(self.estimators_) == 0:
                    raise ValueError('BaseClassifier in AdaBoostClassifier '
                                     'ensemble is worse than random, ensemble '
                                     'can not be fit.')
            self.sample_weight /= np.sum(self.sample_weight)
            return None

        beta = estimator_error / (1. - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1. / beta)

        self.sample_weight *= np.power(
                beta,
                (1. - error_vect) * self.learning_rate)

        self.sample_weight /= np.sum(self.sample_weight)

        return estimator

class AdaBoostClassifier(ISLEBaseEnsembleClassifier):
    """docstring for AdaBoostClassifier"""
    def __init__(self, 
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 alpha=0.1,
                 n_estimators=10,
                 random_state=None,
                 post_model=Lasso(),
                 learning_rate=1.0):
        super(AdaBoostClassifier, self).__init__(
                 base_estimator=base_estimator,
                 alpha=alpha,
                 n_estimators=n_estimators,
                 estimator_params=tuple(),
                 random_state=random_state,
                 bootstrap = True,
                 warm_start=False,
                 post_model=post_model,
                 learning_rate=learning_rate)
    
    def _build_base_estimators(self, estimator, X, y):
        bootstrap_idx = self.pertubater.sample_array(X, self.random_state, self.sample_weight)
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

       # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=self.sample_weight, axis=0))

        n_classes = self.n_classes_

        if estimator_error <= 0:
            # Stop if fit is perfect
            self.sample_weight /= np.sum(self.sample_weight)
            return estimator

        elif estimator_error >= 1. - (1. / n_classes):
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
                if len(self.estimators_) == 0:
                    raise ValueError('BaseClassifier in AdaBoostClassifier '
                                     'ensemble is worse than random, ensemble '
                                     'can not be fit.')
            self.sample_weight /= np.sum(self.sample_weight)
            return None

        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        self.sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((self.sample_weight > 0) |
                                     (estimator_weight < 0)))

        self.sample_weight /= np.sum(self.sample_weight)

        return estimator

class GradientBoostingRegressor(ISLEBaseEnsembleRegressor):
    """docstring for GradientBoostingRegressor"""
    def __init__(self, 
                 loss='ls',
                 base_estimator=DecisionTreeRegressor(),
                 alpha=0.1,
                 n_estimators=10,
                 criterion="friedman_mse",
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 random_state=None,
                 bootstrap=False,
                 warm_start=False,
                 post_model=Lasso(),
                 learning_rate=0.1):
        super(GradientBoostingRegressor, self).__init__(
                 base_estimator=base_estimator,
                 alpha=alpha,
                 n_estimators=n_estimators,
                 estimator_params=("criterion", "max_depth", "min_samples_split",
                                  "min_samples_leaf", "min_weight_fraction_leaf",
                                  "max_features", "max_leaf_nodes", "min_impurity_split",
                                  "random_state"),
                 random_state=random_state,
                 bootstrap=bootstrap,
                 warm_start=warm_start,
                 post_model=post_model,
                 learning_rate=learning_rate)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
        self.sample_mask = None
        self.y_pred = None
        self.X_idx_sorted = None
        self.loss_ = LOSS_FUNCTIONS[loss](1)

    def _build_base_estimators(self, estimator, X, y):
        if self.y_pred is None:
            self.init_ = self.loss_.init_estimator()
            self.init_.fit(X, y)
            self.y_pred = self.init_.predict(X)

        if self.sample_weight is None:
            self.sample_weight = np.ones(X.shape[0], dtype=np.float32)
        
        if self.X_idx_sorted is None:
            self.X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                                 dtype=np.int32)

        if self.sample_mask is None:
            self.sample_mask = np.ones((X.shape[0], ), dtype=np.bool)

        residual = self.loss_.negative_gradient(y, self.y_pred)
        estimator.fit(X, residual, sample_weight=self.sample_weight, check_input=False, X_idx_sorted=self.X_idx_sorted)
          
        self.loss_.update_terminal_regions(estimator.tree_, X, y, residual, self.y_pred, sample_weight=self.sample_weight, sample_mask=self.sample_mask, learning_rate=self.learning_rate)

        return estimator
    
class GradientBoostingClassifier(ISLEBaseEnsembleClassifier):
    """docstring for GradientBoostingClassifier"""
    
    _SUPPORTED_LOSS = ('deviance', 'exponential')

    def __init__(self, 
                 loss='deviance',
                 base_estimator=DecisionTreeRegressor(),
                 alpha=0.1,
                 n_estimators=10,
                 criterion="friedman_mse",
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 random_state=None,
                 bootstrap=False,
                 warm_start=False,
                 post_model=Lasso(),
                 learning_rate=0.1):
        super(GradientBoostingClassifier, self).__init__(
                 base_estimator=base_estimator,
                 alpha=alpha,
                 n_estimators=n_estimators,
                 estimator_params=("criterion", "max_depth", "min_samples_split",
                                  "min_samples_leaf", "min_weight_fraction_leaf",
                                  "max_features", "max_leaf_nodes", "min_impurity_split",
                                  "random_state"),
                 random_state=random_state,
                 bootstrap=bootstrap,
                 warm_start=warm_start,
                 post_model=post_model,
                 learning_rate=learning_rate)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
        self.sample_mask = None
        self.y_pred = None
        self.X_idx_sorted = None
        self.loss = loss
        self.loss_ = None

    def _build_base_estimators(self, estimator, X, y):
        if self.y_pred is None:
            self.init_ = self.loss_.init_estimator()
            self.init_.fit(X, y)
            self.y_pred = self.init_.predict(X)

        original_y = y
        loss = self.loss_

        estimators = [self._make_estimator(append=False, random_state=self.random_state) for _ in xrange(loss.K)]

        if self.sample_weight is None:
            self.sample_weight = np.ones(X.shape[0], dtype=np.float32)
        
        if self.X_idx_sorted is None:
            self.X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                                 dtype=np.int32)

        if self.sample_mask is None:
            self.sample_mask = np.ones((X.shape[0], ), dtype=np.bool)
        
        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y, self.y_pred, k=k,
                                              sample_weight=self.sample_weight)

            estimators[k].fit(X, residual, sample_weight=self.sample_weight, check_input=False, X_idx_sorted=self.X_idx_sorted)
          
            loss.update_terminal_regions(estimators[k].tree_, X, y, residual, self.y_pred, sample_weight=self.sample_weight, sample_mask=self.sample_mask, learning_rate=self.learning_rate)


        return estimators

    def init_stage(self):
        self.estimators_ = []
        self._check_params()
    
    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS
                or self.loss not in LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if self.loss == 'deviance':
            loss_class = (MultinomialDeviance
                          if self.n_classes_ > 2
                          else BinomialDeviance)
        else:
            loss_class = LOSS_FUNCTIONS[self.loss]

        self.loss_ = loss_class(self.n_classes_)

    def _tree_predict_proba(self, estimator_id, X):
        score = np.zeros((X.shape[0], self.loss_.K))
        estimators = self.estimators_[estimator_id]
        for i in xrange(len(estimators)):
            if self.coef_ is None or self.coef_[estimator_id][i] != 0.:
                score[:, i] = estimators[i].predict(X) * self.learning_rate
        # print score[:1,:]
        return self.loss_._score_to_proba(score)

    # def _post_process(self, X, y, check_input=False):
    #     if check_input:
    #         X, y = check_X_y(X, y)

    #     n_samples = X.shape[0]
    #     y_matrix = np.empty((n_samples, self.n_classes_, self.n_estimators))
    #     for i in xrange(self.n_estimators):
    #         y_matrix[:,:,i] = self._tree_predict_proba(i, X)

    #     self.coef_ = np.empty((self.n_estimators, self.n_classes_))
    #     self.intercept_ = np.empty((self.n_classes_,))
        
    #     # self.coef_ = np.ones((self.n_estimators,))
    #     self.post_model.set_params(alpha=self.alpha)
    #     for k in xrange(self.n_classes_):
    #         self.post_model.fit(y_matrix[:,k,:], y)
    #         self.coef_[:,k] = self.post_model.coef_
    #         self.intercept_[k] = self.post_model.intercept_

    #     self.coef_[np.abs(self.coef_) < EPS] = 0.0
    #     print self.coef_
    #     print self.intercept_
    #     # self.coef_ = np.ones((self.n_estimators, self.n_classes_))
    #     # self.intercept_ = 0.
    #     # print self.coef_
    #     return self

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

    # clfs = [clf, clf2]
    # cv = 2
    # for i in xrange(cv):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/float(cv))
        
    #     for j in xrange(len(clfs)):
    #         clfs[j].fit(X_train,y_train)
    #     for j in xrange(len(clfs)):
    #         pred = clfs[j].predict(X_test)
    #         print pred[:10]
    #     print 
    # clf.fit(X_train, y_train)
    # clf2.fit(X_train, y_train)
    # print y_test[:5]
    # print
    # print clf.predict(X_test)[:5]
    # print
    # print clf2.predict(X_test)[:5]
    # print cv.group_cross_validaiton([clf, clf2], X, y, scoring='r2')
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

    # clfs = [clf, clf2]
    # cv = 2
    # for i in xrange(cv):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/float(cv))
        
    #     for j in xrange(len(clfs)):
    #         clfs[j].fit(X_train,y_train)
    #     for j in xrange(len(clfs)):
    #         pred = clfs[j].predict(X_test)
    #         print pred[:10]
    #     print 
    # clf.fit(X_train, y_train)
    # print
    # clf2.fit(X_train, y_train)
    # print y_test[:20]
    # print
    # print clf.predict(X_test)[:20]
    # print
    # print clf2.predict(X_test)[:20]
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

test_GradientBoostingClassifier()