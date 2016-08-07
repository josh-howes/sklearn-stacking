"""
Stacked Generalization (Stacking) Model.

This module contains a Stacked Generalization regressor for
regression estimators.

"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.externals import six
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.externals.joblib import Parallel, delayed


def _parallel_fit(estimator, X, y):
    fitted_estimator = clone(estimator).fit(X, y)
    return fitted_estimator


class StackingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """Stacked Generalization (Stacking) Model.

    Parameters
    ----------
    estimators : list of regressors
        Invoking the ``fit`` method on the ``StackingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        `self.estimators_`.

    combiner : regressor or None, optional (default=None)
        The base regresor to fit on a test train split of the estimators predictions.
        If None, then the base estimator is a linear regressor model.

    cross_val_test_size : float, optional (default=0.33)

    random_state : int or None, optional (default=None)

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators.

    combiner_ : regressor
        Fitted combining regressor.
    """

    def __init__(self, estimators, combiner=None, cross_val_test_size=0.33, random_state=None):

        self.estimators = estimators
        self.combiner = combiner
        self.cross_val_test_size = cross_val_test_size
        self.random_state = random_state


    def fit(self, X, y):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of estimators.')

        if self.combiner is None:
            self.combiner = LinearRegression(fit_intercept=False, normalize=True)

        if not isinstance(self.combiner, RegressorMixin):
            raise AttributeError('Invalid `combiner` attribute, `combiner`'
                                 ' should be an instance of a regressor.')

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=self.cross_val_test_size,
                                                            random_state=self.random_state)

        n_estimators = len(self.estimators)
        self.estimators_ = Parallel(n_jobs=-n_estimators)(
            delayed(_parallel_fit)(
                estimator,
                X_train,
                y_train
            )
            for estimator in self.estimators)

        X_stack = np.asarray([estimator.predict(X_test) for estimator in self.estimators_]).T
        self.combiner_ = self.combiner.fit(X_stack, y_test)

        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        predictions : array-like, shape = [n_samples]
            Regressed values
        """

        check_is_fitted(self, 'estimators_')
        check_is_fitted(self, 'combiner_')

        X_stack = np.asarray([estimator.predict(X) for estimator in self.estimators_]).T
        return self.combiner_.predict(X_stack)


    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
          array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.
        """
        check_is_fitted(self, 'estimators_')
        check_is_fitted(self, 'combiner_')
        return self.predict(X)

    def get_params(self, deep=False):
        """Return estimator parameter names for GridSearch support"""

        if not deep:
            return super(StackingRegressor, self).get_params(deep=False)
        else:
            # TODO: this will not work, need to implement `named_estimators`
            raise NotImplementedError("`deep` attribute not yet supported.")
            out = super(StackingRegressor, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            for name, step in six.iteritems(self.named_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
