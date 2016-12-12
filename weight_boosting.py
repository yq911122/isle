import numpy as np

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils import check_array, check_X_y
from sklearn.linear_model import Lasso

from base import PertubationSampler, ISLEBaseEnsembleRegressor, ISLEBaseEnsembleClassifier
        
class AdaBoostRegressor(ISLEBaseEnsembleRegressor):
    """An AdaBoost regressor.
   
    This class implements the algorithm known as AdaBoost.R2 [1].

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeRegressor)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required.

    alpha: float, optional (default=0.1)
        Constant that multiplies the L1 term.

    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    loss : {'linear', 'square', 'exponential'}, optional (default='linear')
        The loss function to use when updating the weights after each
        boosting iteration.
    
    random_state : RandomState instance or None, optional (default=None)
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each regressor by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    post_model : object, optional (default=Lasso)
        The L1 regularized regression model

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``base_estimator``.

    References
    ----------
    .. [1] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.
    """
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
    """An AdaBoost classifier.

    This class implements the algorithm known as AdaBoost-SAMME [1].

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.

    alpha: float, optional (default=0.1)
        Constant that multiplies the L1 term for post-processing.

    n_estimators : integer, optional (default=10)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    post_model : object, optional (default=Lasso)
        The L1 regularized regression model 

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes]
        The classes labels.
    
    n_classes_ : int
        The number of classes.

    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``base_estimator``.

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
    """
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