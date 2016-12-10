import warnings
from warnings import warn

import numpy as np

# from scipy.sparse import issparse
from sklearn.ensemble._gradient_boosting import predict_stages

from sklearn.ensemble.gradient_boosting import LeastSquaresError, BinomialDeviance, MultinomialDeviance, ExponentialLoss

from sklearn.utils import check_array, check_X_y, check_random_state, compute_sample_weight
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted

from base import PertubationSampler, ISLEBaseEnsembleRegressor, ISLEBaseEnsembleClassifier

from sklearn.tree._tree import DTYPE, DOUBLE
LOSS_FUNCTIONS = {'ls': LeastSquaresError,
                  'deviance': None,
                  'exponential': ExponentialLoss
                  }


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