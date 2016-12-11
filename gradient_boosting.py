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
    """Gradient Boosting for regression.
    
    Parameters
    ----------
    loss : {'ls'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).
    
    base_estimator : object, optional (default=DecisionTreeRegressor)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required.

    alpha: float, optional (default=0.1)
        Constant that multiplies the L1 term for post-processing.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.
    
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.
    
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
   
    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
 
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
 
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
 
    min_impurity_split : float, optional (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
        .. versionadded:: 0.18
 
    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.
 
    init : BaseEstimator, None, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.
 
    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.
 
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    post_model : object, optional (default=Lasso)
        The L1 regularized regression model
 
    Attributes
    ----------
    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, 1]
        The collection of fitted sub-estimators.

    """
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
        """private method to fit the single estimator in the ensemble

        Parameters
        ----------
        estimator : object, the single estimator for fitting

        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            the training response

        Returns
        -------
        estimator: object, fitted single estimator

        """
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
    """Gradient Boosting for Classification.
    
    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.
    
    base_estimator : object, optional (default=DecisionTreeRegressor)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required.

    alpha: float, optional (default=0.1)
        Constant that multiplies the L1 term for post-processing.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.
    
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.
    
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
   
    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
 
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
 
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
 
    min_impurity_split : float, optional (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
        .. versionadded:: 0.18
 
    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.
 
    init : BaseEstimator, None, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.
 
    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.
 
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.
        
    post_model : object, optional (default=Lasso)
        The L1 regularized regression model
 
    Attributes
    ----------
    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, 1]
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    """
    
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
        """private method to fit the single estimator in the ensemble

        Parameters
        ----------
        estimator : object, the single estimator for fitting

        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            the training response

        Returns
        -------
        estimator: object, fitted single estimator

        """    
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
        """init fitting process. Estimators will be cleared."""
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
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the ensemble. The
        class probability of a single tree is the fraction of samples of the same
        class in a leaf.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        score = np.zeros((X.shape[0], self.loss_.K))
        estimators = self.estimators_[estimator_id]
        for i in xrange(len(estimators)):
            if self.coef_ is None or self.coef_[estimator_id][i] != 0.:
                score[:, i] = estimators[i].predict(X) * self.learning_rate
        # print score[:1,:]
        return self.loss_._score_to_proba(score)