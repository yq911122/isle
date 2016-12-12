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
    """class that perfroms perturbation sampling.

    For more details about perturbation sampling, check Friedman(2003)"""
    def __init__(self):
        super(PertubationSampler, self).__init__()

    def _random_function(self, random_state):
        """private function that generates a random function. This function is 
        usually added on the loss function in order to pertubate sampling.

        Parameters
        ----------
        random_state: numpy.random.RandomState

        Returns
        -------
        random_func: random_state.rand

        """
        return random_state.rand
    
    def pertube_distribution_weight(self, random_state, n_samples, multipler):
        """generate sample weights [w_m(z)]^r so as to modify the sample distribution.

                q_m(z) = q(z) * [w_m(z)]^r
        
        Parameters
        ----------
        random_state: numpy.random.RandomState

        n_samples: int. 
                   Number of sample weights that will be generated
        
        multipler: float. 
                   r in the equation

        Returns
        -------
        sample weight: numpy array, shape = [n_samples]

        """
        curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        indices = _generate_sample_indices(random_state, n_samples)
        sample_counts = bincount(indices, minlength=n_samples)
        curr_sample_weight *= np.power(sample_counts, multipler)
        return curr_sample_weight

    def pertube_loss_function(self, X, random_state, multipler, function=None):
        """calculate add-on value of X so as to pertube loss function value. It is
        calculated as:

            r * g(X)
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        random_state: numpy.random.RandomState
        
        multipler: float. r in the equation

        function: python function with X as input and return array-like of shape = [n_samples].
            g() in the equation. If None, a random function will be provided.

        Returns
        -------
        add-on value: numpy array, shape = [n_samples]

        """
        if not function:
            function = self._random_function(random_state)
        return function(X) * multipler
        
    def sample_X_y(self, X, y, frac=1.0, replace=True, return_index=False):
        """sample X and y
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            the training response

        frac: float. Fraction of n_samples to be sampled
        
        replace: bool. If true, apply sampling with replacement 

        return_index: bool. If true, return sample index as 
            array-like of shape = [n_samples * frac]

        Returns
        -------
        X_sample: array-like of shape = [n_samples * frac, n_features]
        
        y_sample: array-like of shape = [n_samples * frac]

        sample_index: array-like of shape = [n_samples * frac]. Return if return_index = True

        """        
        X, y = check_X_y(X, y)
        n_population = X.shape[0]
        idx = np.random.choice(n_population, n_population*frac, replace=replace)
        if return_index:
            return X[idx], y[idx], idx
        return X[idx], y[idx]

    def sample_features(self, X, mode='auto', return_index=False):
        """sample features of X
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        mode : string. The number of features to consider when looking for the best split:
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        return_index: bool. If true, return sample index as 
            array-like of shape = [n_features * frac]

        Returns
        -------
        X_sample: array-like of shape = [n_samples, n_features * frac]
        
        sample_feature_index: array-like of shape = [n_features * frac]. Return if return_index = True

        """     

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
        """apply boostrapping on X
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        random_state: numpy.random.RandomState

        sample_weight: array-like of shape = [n_samples].
            probability of X to be chosen in boostrap process.

        Returns
        -------
        bootstrap_idx: array-like of shape = [n_samples].
            array index generated by bootstrapping.

        """  
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
    """Base class for ISLE (Importance Sampled Learning Ensembles)
    
    Warning: This class should not be used directly. Use derived classes
    instead.
    """
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

        self.estimators_ = []

    def fit(self, X, y):
        """Build a ensemble from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """

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
        """init fitting process. Estimators will be cleared."""

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

    # def _pertube_sampling(self, X, y):
    #     pass

    def _build_base_estimators(self, estimator, X, y):
        pass
  
    def _validate_y_class(self, y):
        # Default implementation
        return y

class ISLEBaseEnsembleRegressor(ISLEBaseEnsemble):
    """Base class for ISLE (Importance Sampled Learning Ensembles) Regressors
    
    Warning: This class should not be used directly. Use derived classes
    instead.
    """
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
        """post-process ensemble by Lasso method so as to gain a more parsimonious model.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        check_input : boolean, (default=True)
            Allow to bypass several input checking.

        Returns
        -------
        self : object
            Returns self.

        """
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
        """Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """

        X = check_array(X)

        y_hat = np.zeros((X.shape[0],))
        for i, c in zip(xrange(self.n_estimators), self.coef_):
            if c != 0.0:
                y_hat += c * self._tree_predict(i, X)

        return y_hat + self.intercept_

    def _tree_predict(self, estimator_id, X):
        """Predict regression value for X by a estimator in the ensemble.

        Parameters
        ----------
        estimator_id : integer
            the ith estimator of ensemble self.estimators_

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        return self.estimators_[estimator_id].predict(X) * self.learning_rate

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise Error("Estimator not fitted, "
                        "call `fit` before `feature_importances_`.")

        all_importances = self.estimators_[0].feature_importances_ * self.coef_[0]
        for i in xrange(1, len(self.estimators_)):
            all_importances += self.estimators_[i].feature_importances_ * self.coef_[i]

        return all_importances / np.sum(all_importances)

class ISLEBaseEnsembleClassifier(ISLEBaseEnsemble):
    """Base class for ISLE (Importance Sampled Learning Ensembles) Regressors
    
    Warning: This class should not be used directly. Use derived classes
    instead.
    """
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
        """post-process ensemble by Lasso method so as to gain a more parsimonious model.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        check_input : boolean, (default=True)
            Allow to bypass several input checking.

        Returns
        -------
        self : object
            Returns self.

        """
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

    def predict(self, X, check_input=True):
        """Predict class for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        if check_input:
            X = check_array(X)
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0) 

    def predict_proba(self, X):
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
        """Predict class probabilities for X by a estimator in the ensemble.

        Parameters
        ----------
        estimator_id : integer
            the ith estimator of ensemble self.estimators_

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        return self.estimators_[estimator_id].predict_proba(X) * self.learning_rate

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise Error("Estimator not fitted, "
                        "call `fit` before `feature_importances_`.")

        all_importances = self.estimators_[0].feature_importances_ * np.sum(self.coef_[0])
        for estimator in self.estimators_[1:]:
            all_importances += self.estimators_[i].feature_importances_ * np.sum(self.coef_[i])

        return all_importances / len(self.estimators_)