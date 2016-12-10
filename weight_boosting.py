import numpy as np

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils import check_array, check_X_y
from sklearn.linear_model import Lasso

from base import PertubationSampler, ISLEBaseEnsembleRegressor, ISLEBaseEnsembleClassifier

        
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