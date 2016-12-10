import warnings
from warnings import warn

import numpy as np
import numbers

from sklearn.linear_model import Lasso
from sklearn.base import clone, BaseEstimator
from sklearn.utils.fixes import bincount

from sklearn.utils import check_random_state, check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets

from sklearn.tree._tree import DTYPE, DOUBLE



def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices

MAX_RAND_SEED = np.iinfo(np.int32).max
EPS = 1e-10

def _set_random_states(estimator, random_state=None):
    """Sets fixed random_state parameters for an estimator
    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.
    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.
    random_state : numpy.RandomState or int, optional
        Random state used to generate integer values.
    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:
        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == 'random_state' or key.endswith('__random_state'):
            to_set[key] = random_state.randint(MAX_RAND_SEED)

    if to_set:
        estimator.set_params(**to_set)

class PertubationSampler(object):
    """docstring for PertubationSampler"""
    def __init__(self):
        super(PertubationSampler, self).__init__()

    def _random_function(self, random_state):
        return random_state.rand
    
    def pertube_distribution_weight(self, random_state, n_samples, multipler):
        curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        indices = _generate_sample_indices(random_state, n_samples)
        sample_counts = bincount(indices, minlength=n_samples)
        curr_sample_weight *= np.power(sample_counts, multipler)
        return curr_sample_weight

    def pertube_loss_function(self, X, random_state, multipler, function=None):
        if not function:
            function = self._random_function(random_state)
        return function(X) * multipler
        
    def sample_X_y(self, X, y, frac=1.0, replace=True, return_index=False):
        X, y = check_X_y(X, y)
        n_population = X.shape[0]
        idx = np.random.choice(n_population, n_population*frac, replace=replace)
        if return_index:
            return X[idx], y[idx], idx
        return X[idx], y[idx]

    def sample_features(self, X, mode='auto', return_index=False):
        X = check_array(X)
        n_features = X.shape[1]
        if mode in ['auto', 'sqrt']:
            sample_features = int(np.sqrt(n_features))
        elif mode == 'log2':
            sample_features = int(np.log(n_features))
        elif not mode:
            sample_features = n_features
        idx = np.random.choice(n_features, sample_features, replace=False)
        if return_index:
            return X[:, idx], idx
        return X[:, idx]

    def sample_array(self, X, random_state=None, sample_weight=None):
        if random_state is None:
            random_state = check_random_state(random_state)

        X = check_array(X)
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0], ))
        cdf = sample_weight.cumsum()
        cdf /= cdf[-1]
        uniform_samples = random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)
        return bootstrap_idx
        
class ISLEBaseEnsemble(BaseEstimator):
    """docstring for ISLEBaseEnsemble"""
    def __init__(self,  
                 base_estimator, 
                 alpha = 0.1,
                 n_estimators=10, 
                 estimator_params=tuple(),
                 post_model=Lasso(),
                 random_state=None,
                 warm_start=False,
                 bootstrap=True,
                 learning_rate=1.0):
        super(ISLEBaseEnsemble, self).__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0
        self.post_model = post_model
        self.pertubater = PertubationSampler()
        self.random_state = random_state
        self.warm_start = warm_start
        self.bootstrap = bootstrap
        self.learning_rate = learning_rate
        self.sample_weight = None
        self.classses_ = 0

        self.estimators_ = []

    def fit(self, X, y):
        # check input

        X, y = check_X_y(X, y, dtype=DTYPE)
        y = self._validate_y_class(y)
        
        # self.n_outputs_ = y.shape[1]
        n_samples, self.n_features_ = X.shape

        if self.sample_weight is None:
            self.sample_weight = np.ones((n_samples,))

        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start:
            self.init_stage()

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")

        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = []
            for i in xrange(n_more_estimators):
                tree = self._make_estimator(append=False,
                                            random_state=random_state)
                trees.append(tree)

            for i in xrange(self.n_estimators):
                tree = self._build_base_estimators(trees[i], X, y)
                if tree is None:
                    warn("cannot fit %d estimators. %d esitmators are fitted." % (self.n_estimators, i+1))
                    break
                trees[i] = tree
            self.estimators_.extend(trees[:i+1])

        return self._post_process(X, y)

    def init_stage(self):
        self.estimators_ = []

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**dict((p, getattr(self, p))
                                    for p in self.estimator_params))
        # print estimator.get_params()

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _pertube_sampling(self, X, y):
        pass

    def _build_base_estimators(self, estimator, X, y):
        pass
  
    def _validate_y_class(self, y):
        return y

class ISLEBaseEnsembleRegressor(ISLEBaseEnsemble):
    """docstring for ISLEBaseEnsembleRegressor"""
    def __init__(self,  
                 base_estimator, 
                 alpha = 0.1,
                 n_estimators=10, 
                 estimator_params=tuple(),
                 post_model=Lasso(),
                 random_state=None,
                 warm_start=False,
                 bootstrap=True,
                 learning_rate=1.0):
        super(ISLEBaseEnsembleRegressor, self).__init__(
            base_estimator=base_estimator,
            alpha=alpha,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap = bootstrap,
            warm_start=warm_start,
            post_model=post_model,
            random_state=random_state,
            learning_rate=learning_rate)

    def _post_process(self, X, y, check_input=False):
        if check_input:
            X, y = check_X_y(X, y)

        n_samples = X.shape[0]
        y_matrix = np.empty((n_samples, self.n_estimators))
        for i in xrange(self.n_estimators):
            y_matrix[:,i] = self._tree_predict(i, X)

        self.post_model.set_params(alpha=self.alpha)
        self.post_model.fit(y_matrix, y)
        self.coef_ = self.post_model.coef_
        self.coef_[np.abs(self.coef_) < EPS] = 0.0
        self.intercept_ = self.post_model.intercept_

        return self

    def predict(self, X):
        X = check_array(X)

        y_hat = np.zeros((X.shape[0],))
        for i, c in zip(xrange(self.n_estimators), self.coef_):
            if c != 0.0:
                y_hat += c * self._tree_predict(i, X)

        return y_hat + self.intercept_

    def _tree_predict(self, estimator_id, X):
        return self.estimators_[estimator_id].predict(X) * self.learning_rate

class ISLEBaseEnsembleClassifier(ISLEBaseEnsemble):
    """docstring for ISLEBaseEnsembleClassifier"""
    def __init__(self,  
                 base_estimator, 
                 alpha = 0.1,
                 n_estimators=10, 
                 estimator_params=tuple(),
                 post_model=Lasso(),
                 random_state=None,
                 warm_start=False,
                 bootstrap=True,
                 learning_rate=1.0):
        super(ISLEBaseEnsembleClassifier, self).__init__(
            base_estimator=base_estimator,
            alpha=alpha,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap = bootstrap,
            warm_start=warm_start,
            post_model=post_model,
            learning_rate=learning_rate)

        self.n_classes_ = 0
        self.classes_ = None

    def _post_process(self, X, y, check_input=False):
        if check_input:
            X, y = check_X_y(X, y)

        n_samples = X.shape[0]
        y_matrix = np.empty((n_samples, self.n_classes_, self.n_estimators))
        original_y = y

        for i in xrange(self.n_estimators):
            y_matrix[:,:,i] = self._tree_predict_proba(i, X)

        self.coef_ = np.empty((self.n_estimators, self.n_classes_))
        self.intercept_ = np.empty((self.n_classes_,))
        
        self.post_model.set_params(alpha=self.alpha)
        for k in xrange(self.n_classes_):
            if self.n_classes_ > 2:
                y = np.array(original_y == k, dtype=np.float64)
            self.post_model.fit(y_matrix[:,k,:], y)
            self.coef_[:,k] = self.post_model.coef_
            self.intercept_[k] = self.post_model.intercept_

        self.coef_[np.abs(self.coef_) < EPS] = 0.0

        return self

    def predict(self, X):
        X = check_array(X)
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0) 

    def predict_proba(self, X):
        X = check_array(X)
        proba = np.zeros((X.shape[0], self.n_classes_))
        for i in xrange(self.n_estimators):
            proba += self.coef_[i] * self._tree_predict_proba(i, X)
        proba += self.intercept_
        return proba

    def _validate_y_class(self, y):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y

    def _tree_predict_proba(self, estimator_id, X):
        return self.estimators_[estimator_id].predict_proba(X) * self.learning_rate
