import numpy as np

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Lasso

from base import PertubationSampler, ISLEBaseEnsembleRegressor, ISLEBaseEnsembleClassifier


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