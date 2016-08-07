import sys
import unittest
from nose.tools import assert_raises, assert_equal, assert_true
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import NotFittedError
from ensemble.stacking_regressor import StackingRegressor


class TestStackingRegressor(unittest.TestCase):

    def setUp(self):
        self.dataset = fetch_california_housing()

    def test_is_fitted_predict(self):
        """A unfitted model throws an exception if used to `predict`."""
        estimators = [LinearRegression(), LinearRegression()]
        stacker = StackingRegressor(estimators=estimators)

        with assert_raises(NotFittedError):
            stacker.predict(self.dataset.data)

    def test_is_fitted_transform(self):
        """A unfitted model throws an exception if used to `transform`."""
        estimators = [LinearRegression(), LinearRegression()]
        stacker = StackingRegressor(estimators=estimators)

        with assert_raises(NotFittedError):
            stacker.transform(self.dataset.data)

    def test_combiner_is_regressor(self):
        """Throw and error when something other than a regressor is a `combiner`."""
        from sklearn.cluster import KMeans
        estimators = [LinearRegression(), LinearRegression()]
        stacker = StackingRegressor(estimators=estimators, combiner=KMeans())
        with assert_raises(AttributeError):
            stacker.fit(self.dataset.data, self.dataset.target)

    def test_custom_cobminer(self):
        """Allow the use of a different `combiner`."""
        from sklearn.tree import DecisionTreeRegressor
        estimators = [LinearRegression(), LinearRegression()]
        stacker = StackingRegressor(estimators=estimators, combiner=DecisionTreeRegressor())
        stacker.fit(self.dataset.data, self.dataset.target)
        pred = stacker.predict(self.dataset.data)
        assert_equal(pred.shape, self.dataset.target.shape)

    @unittest.skip("Unsure how best to test this yet.")
    def test_combiner_coef_toyproblem(self):
        """Ensure toy problem's `combiner` coefficients are correct."""
        estimators = [LinearRegression(), LinearRegression()]
        stacker = StackingRegressor(estimators=estimators)
        stacker.fit(self.dataset.data, self.dataset.target)

        stacker_preds = stacker.predict(self.dataset.data)
        linear_preds = stacker.estimators_[0].predict(self.dataset.data)

        assert_true((stacker_preds == linear_preds).all())

    def test_getparams_shallow(self):
        """Retrieve the correct params."""
        estimators = [LinearRegression(), LinearRegression()]
        stacker = StackingRegressor(estimators=estimators)
        params = stacker.get_params(deep=False)

        desired_params = {'random_state': None,
                          'cross_val_test_size': 0.33,
                          'combiner': None,
                          'estimators': [
                              LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False),
                              LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)]}

        assert_equal(str([(k, v) for k, v in sorted(params.items())]),
                     str([(k, v) for k, v in sorted(desired_params.items())]))

    def test_getparams_deep(self):
        """Retrieve the correct params."""
        estimators = [LinearRegression(), LinearRegression()]
        stacker = StackingRegressor(estimators=estimators)
        with assert_raises(NotImplementedError):
            stacker.get_params(deep=True)


if __name__ == '__main__':
    sys.exit(unittest.main())
