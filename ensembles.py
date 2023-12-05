import numpy as np
from numpy import ndarray
from scipy.optimize import minimize_scalar
# from sklearn._typing import ArrayLike, MatrixLike
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        random_state=42, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.

        random_state : int
            set the random state for estimators. 42 by default
        """
        self.trees = None

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        if isinstance(feature_subsample_size, float) and \
                (feature_subsample_size > 1.0 or feature_subsample_size < 0.0):
            raise ValueError(
                'feature_subsample_size must be in range [0,1] or be integer')
        self.fss = feature_subsample_size if feature_subsample_size is not None else 1/3

        self.tree_parameters = trees_parameters

    def fit(self, X: np.ndarray, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """

        self.trees = []

        if (isinstance(self.fss, int) and self.fss > X.shape[1]):
            raise ValueError(
                'X have less features than expected by feature_subsample_size')

        fss = self.fss if isinstance(
            self.fss, int) else round(X.shape[1] * self.fss)

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                criterion='squared_error',
                splitter='random',
                max_depth=self.max_depth,
                max_features=fss,
                random_state=self.random_state,
                **self.tree_parameters
            )
            tree.fit(X, y)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        if self.trees is None:
            raise ValueError('model is not fited')

        preds = [tree.predict(X) for tree in self.trees]
        return np.mean(np.array(preds), axis=1)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5,
        feature_subsample_size=None, random_state=42,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float | None
            The size of feature set for each tree. If None then use one-third of all features.

        random_state : int
            set the random state for estimators. 42 by default
        """
        self.trees = None

        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        if isinstance(feature_subsample_size, float) and \
                (feature_subsample_size > 1.0 or feature_subsample_size < 0.0):
            raise ValueError(
                'feature_subsample_size must be in range [0,1] or be integer')
        self.fss = feature_subsample_size if feature_subsample_size is not None else 1/3
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        self.trees = []

        if (isinstance(self.fss, int) and self.fss > X.shape[1]):
            raise ValueError(
                'X have less features than expected by feature_subsample_size')

        fss = self.fss if isinstance(
            self.fss, int) else round(X.shape[1] * self.fss)

        grad = y

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                criterion='squared_error',
                splitter='best',
                max_features=fss,
                random_state=self.random_state,
                max_depth=self.max_depth
            )
            tree.fit(X, grad)
            self.trees.append(tree)

            grad = self.lr * (y - self.predict(X))

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        if self.trees is None:
            raise ValueError('model is not fited')

        preds = [tree.predict(X) for tree in self.trees]
        return np.sum(np.array(preds), axis=1)
