import warnings
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from pandas import DataFrame, Series
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# suppress warnings outcoming from sklearn models


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class RandomForestMSE:
    """
        Methods
        -------
        fit(X, y, X_val=None, y_val=None)
            Fit the random forest model to the training data.

        make_metrics(X, y)
            Calculate evaluation metrics for the model predictions.

        predict(X)
            Predict the target variable for the input data.
    """

    def __init__(
        self, n_estimators, *, max_depth=None, feature_subsample_size=None,
        splitter='best', bootstrap=None, random_state=42, **trees_parameters
    ):
        """
        Random Forest for Mean Squared Error (MSE) regression.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.

        max_depth : int, optional
            The maximum depth of the tree. If None, then there is no limit.

        feature_subsample_size : float, optional
            The size of the feature set for each tree. If None, then use one-third of all features.

        splitter : {'best', 'random'}, default='best'
            Criterion for splitting nodes in the trees.

        bootstrap : None, int, or float, default=None
            If None, bootstrapping is not performed. Otherwise, it defines the splits for bootstrapping.

        random_state : int, default=42
            Set the random state for estimators.

        **trees_parameters : dict
            Additional parameters to be passed to the underlying DecisionTreeRegressor.

        Raises
        ------
        ValueError
            If `feature_subsample_size` is not in the valid range of values.

        """
        self.trees = None

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.splitter = splitter
        self.random_state = random_state
        self.bootstrap = bootstrap

        # check if feature_subsample_size is in valid range of values
        if isinstance(feature_subsample_size, float) and \
                (feature_subsample_size > 1.0 or feature_subsample_size < 0.0):
            raise ValueError(
                'feature_subsample_size must be in range [0,1] or be an integer')
        self.fss = feature_subsample_size if feature_subsample_size is not None else 1/3

        self.tree_parameters = trees_parameters

    def fit(self, X: np.ndarray, y, X_val=None, y_val=None):
        """
        Fit the Random Forest model to the training data.

        Parameters
        ----------
        X : numpy ndarray
            Array of size (n_objects, n_features) representing the training input samples.

        y : numpy ndarray
            Array of size (n_objects) representing the target values.

        X_val : numpy ndarray, optional
            Array of size (n_val_objects, n_features) representing the validation input samples.

        y_val : numpy ndarray, optional
            Array of size (n_val_objects) representing the validation target values.

        Returns
        -------
        self : GradientBoostingMSE
            The fitted RandomForestMSE object if history is not required.

        history : dict
            The history of evaluation metrics during training, if X_val and y_val are provided. Otherwise, None.
            The dictionary has the following structure:
            {
                'rmse': {'train': [], 'test': []},
                'r2': {'train': [], 'test': []},
                'mape': {'train': [], 'test': []},
                'mae': {'train': [], 'test': []}
            }

        Raises
        ------
        ValueError
            If `X` or `y` are not numpy data structures or if `X` has fewer features than expected by `feature_subsample_size`.

        """
        self.trees = []

        # if X or y are not numpy data structures, redefine them
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        if isinstance(y, Series):
            y = y.to_numpy()

        # generator, used in trees random_state
        rnd_gen = np.random.Generator(np.random.PCG64(self.random_state))

        # check if predefined fss is less than count of features in objects
        if (isinstance(self.fss, int) and self.fss > X.shape[1]):
            raise ValueError(
                'X has fewer features than expected by feature_subsample_size')

        # if bootstrapping is enabled, check and prepare splitting point
        bootstrap = None
        if self.bootstrap is not None:
            if (isinstance(self.bootstrap, int) and self.bootstrap > X.shape[0] or
                    isinstance(self.bootstrap, float) and (self.bootstrap > 1.0 or self.bootstrap < 0)):
                raise ValueError('bootstrap index out of range')
            else:
                bootstrap = self.bootstrap if isinstance(
                    self.bootstrap, int) else round(self.bootstrap * X.shape[0])

        history = None
        if X_val is not None and y_val is not None:
            history = {
                'rmse': {'train': [], 'test': []},
                'r2': {'train': [], 'test': []},
                'mape': {'train': [], 'test': []},
                'mae': {'train': [], 'test': []}
            }

        for _ in range(self.n_estimators):
            idx = np.random.permutation(
                X.shape[0])[:bootstrap] if bootstrap is not None else np.arange(X.shape[0])

            # define tree structure
            tree = DecisionTreeRegressor(
                criterion='squared_error',
                splitter=self.splitter,
                max_depth=self.max_depth,
                max_features=self.fss,
                random_state=rnd_gen.integers(0, 100_000_000),
                **self.tree_parameters
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

            if history is not None:
                rmse, mae, r2, mape = self.make_metrics(X, y)
                history['rmse']['train'].append(rmse)
                history['mae']['train'].append(mae)
                history['r2']['train'].append(r2)
                history['mape']['train'].append(mape)

                rmse, mae, r2, mape = self.make_metrics(X_val, y_val)
                history['rmse']['test'].append(rmse)
                history['mae']['test'].append(mae)
                history['r2']['test'].append(r2)
                history['mape']['test'].append(mape)

        return self if history is None else history

    def make_metrics(self, X, y):
        """
        Calculate evaluation metrics for the given input samples and target values.

        Parameters
        ----------
        X : numpy ndarray
            Array of size (n_objects, n_features) representing the input samples.

        y : numpy ndarray
            Array of size (n_objects) representing the target values.

        Returns
        -------
        rmse : float
            Root Mean Squared Error (RMSE) metric.

        mae : float
            Mean Absolute Error (MAE) metric.

        r2 : float
            R-squared (coefficient of determination) metric.

        mape : float
            Mean Absolute Percentage Error (MAPE) metric.

        """
        preds = self.predict(X)

        rmse = mean_squared_error(y, preds, squared=False)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        mape = mean_absolute_percentage_error(y, preds)

        return rmse, mae, r2, mape

    def predict(self, X):
        """
        Predict the target values for the given input samples.

        Parameters
        ----------
        X : numpy ndarray
            Array of size (n_objects, n_features) representing the input samples.

        Returns
        -------
        y : numpy ndarray
            Array of size (n_objects) representing the predicted target values.

        Raises
        ------
        ValueError
            If the model is not fitted yet.

        """
        # if model is not fitted yet, raise an error
        if self.trees is None:
            raise ValueError('Model is not fitted')
        preds = np.array([tree.predict(X) for tree in self.trees])

        return np.mean(preds, axis=0)


class GradientBoostingMSE:
    """
        Methods
        -------
        fit(X, y, X_val=None, y_val=None)
            Fit the gradient boosting model to the training data.

        make_metrics(X, y)
            Calculate evaluation metrics for the model predictions.

        predict(X)
            Predict the target variable for the input data.
    """

    def __init__(
        self, n_estimators, *, learning_rate=0.1, max_depth=5,
        splitter='best', feature_subsample_size=None, random_state=42,
        bootstrap=None,
        **trees_parameters
    ):
        """
        Gradient Boosting for Mean Squared Error (MSE) regression.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float, optional (default=0.1)
            Learning rate shrinks the contribution of each tree.

        max_depth : int, optional (default=5)
            The maximum depth of the tree. If None, there is no limit.

        splitter : {'best', 'random'}, optional (default='best')
            The strategy used to choose the split at each node.

        feature_subsample_size : float or None, optional (default=None)
            The size of the feature set for each tree. If None, one-third of all features are used.

        bootstrap : None, int, or float, optional (default=None)
            If None, bootstrapping is not performed. Otherwise, it defines the splits for bootstrapping.

        random_state : int, optional (default=42)
            Set the random state for estimators.

        trees_parameters : dict
            Additional parameters to be passed to the DecisionTreeRegressor.

        Raises
        ------
        ValueError
            If feature_subsample_size is not in the valid range of values.

        """
        self.weights = None
        self.trees = None

        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.splitter = splitter
        self.bootstrap = bootstrap

        # check if feature_subsample_size is in valid range of values
        if isinstance(feature_subsample_size, float) and \
                (feature_subsample_size > 1.0 or feature_subsample_size < 0.0):
            raise ValueError(
                'feature_subsample_size must be in range [0,1] or be an integer')
        self.fss = feature_subsample_size if feature_subsample_size is not None else 1/3
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the gradient boosting model to the training data.

        Parameters
        ----------
        X : numpy ndarray or pandas DataFrame
            Array of size (n_objects, n_features) or DataFrame containing the input features.

        y : numpy ndarray or pandas Series
            Array of size (n_objects) or Series containing the target variable.

        X_val : numpy ndarray or pandas DataFrame, optional (default=None)
            Array of size (n_objects, n_features) or DataFrame containing the validation input features.

        y_val : numpy ndarray or pandas Series, optional (default=None)
            Array of size (n_objects) or Series containing the validation target variable.

        Raises
        ------
        ValueError
            If X has fewer features than expected by feature_subsample_size.
            If bootstrap index is out of range.

        Returns
        -------
        self : GradientBoostingMSE
            The fitted gradient boosting model if history is not required.

        history : dict or None
            The history of evaluation metrics during training, if X_val and y_val are provided. Otherwise, None.
            The dictionary has the following structure:
            {
                'rmse': {'train': [], 'test': []},
                'r2': {'train': [], 'test': []},
                'mape': {'train': [], 'test': []},
                'mae': {'train': [], 'test': []}
            }
        """
        self.weights = []
        self.trees = []

        # if X or y are not numpy data structures, convert them
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        if isinstance(y, Series):
            y = y.to_numpy()

        # generator used in trees random_state
        rnd_gen = np.random.Generator(np.random.PCG64(self.random_state))

        # check if predefined fss is less than count of features in objects
        if (isinstance(self.fss, int) and self.fss > X.shape[1]):
            raise ValueError(
                'X has fewer features than expected by feature_subsample_size')

        # if bootstrapping is enabled, check and prepare splitting point
        bootstrap = None
        if self.bootstrap is not None:
            if (isinstance(self.bootstrap, int) and self.bootstrap > X.shape[0] or
                    isinstance(self.bootstrap, float) and (self.bootstrap > 1.0 or self.bootstrap < 0)):
                raise ValueError('bootstrap index out of range')
            else:
                bootstrap = self.bootstrap if isinstance(
                    self.bootstrap, int) else round(self.bootstrap * X.shape[0])

        history = None
        if X_val is not None and y_val is not None:
            history = {
                'rmse': {'train': [], 'test': []},
                'r2': {'train': [], 'test': []},
                'mape': {'train': [], 'test': []},
                'mae': {'train': [], 'test': []}
            }

        # initialize first target for boosting
        grad = y.copy()
        for _ in range(self.n_estimators):
            idx = np.random.permutation(
                X.shape[0])[:bootstrap] if bootstrap is not None else np.arange(X.shape[0])

            tree = DecisionTreeRegressor(
                criterion='squared_error',
                splitter=self.splitter,
                max_features=self.fss,
                random_state=rnd_gen.integers(0, 100_000_000),
                max_depth=self.max_depth,
                **self.trees_parameters
            )
            tree = tree.fit(X[idx], grad[idx])
            preds = tree.predict(X)
            alpha = minimize_scalar(fun=lambda a: np.sum(
                (grad - a * preds) ** 2), bounds=(0, 10000)).x

            grad -= self.lr * alpha * preds
            self.trees.append(tree)
            self.weights.append(self.lr * alpha)

            if history is not None:
                rmse, mae, r2, mape = self.make_metrics(X, y)
                history['rmse']['train'].append(rmse)
                history['mae']['train'].append(mae)
                history['r2']['train'].append(r2)
                history['mape']['train'].append(mape)

                rmse, mae, r2, mape = self.make_metrics(X_val, y_val)
                history['rmse']['test'].append(rmse)
                history['mae']['test'].append(mae)
                history['r2']['test'].append(r2)
                history['mape']['test'].append(mape)

        return self if history is None else history

    def make_metrics(self, X, y):
        """
        Calculate evaluation metrics for the model predictions.

        Parameters
        ----------
        X : numpy ndarray or pandas DataFrame
            Array of size (n_objects, n_features) or DataFrame containing the input features.

        y : numpy ndarray or pandas Series
            Array of size (n_objects) or Series containing the target variable.

        Returns
        -------
        rmse : float
            Root Mean Squared Error (RMSE) metric.

        mae : float
            Mean Absolute Error (MAE) metric.

        r2 : float
            R-squared (coefficient of determination) metric.

        mape : float
            Mean Absolute Percentage Error (MAPE) metric.
        """
        preds = self.predict(X)

        rmse = mean_squared_error(y, preds, squared=False)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        mape = mean_absolute_percentage_error(y, preds)

        return rmse, mae, r2, mape

    def predict(self, X):
        """
        Predict the target variable for the input data.

        Parameters
        ----------
        X : numpy ndarray or pandas DataFrame
            Array of size (n_objects, n_features) or DataFrame containing the input features.

        Raises
        ------
        ValueError
            If the model is not fitted yet.

        Returns
        -------
        y : numpy ndarray
            Array of size (n_objects) containing the predicted target variable.
        """
        # if the model is not fitted yet, raise an error
        if self.trees is None or self.weights is None:
            raise ValueError('Model is not fitted')
        preds = np.array([w * tree.predict(X)
                         for w, tree in zip(self.weights, self.trees)])
        return np.sum(preds, axis=0)
