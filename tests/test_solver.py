import unittest

import pandas as pd
import numpy as np

from qtune.solver import ForwardingSolver
from qtune.solver import NewtonSolver
from qtune.solver import make_target
from qtune.gradient import FiniteDifferencesGradientEstimator


class ForwardingSolverTests(unittest.TestCase):
    def test_forwarding(self):
        v2p = pd.Series(['v1', 'v2', 'v3'], index=['p1', 'p2', 'p3'])

        start = pd.Series([10, 11, 12, 13], index=['v1', 'v2', 'v3', 'v4'])

        fs = ForwardingSolver(values_to_position=v2p,
                              current_position=start,
                              target=pd.DataFrame())

        volts = pd.Series([1.1, 2.2, 3.3, 4.4], index=start.index)
        params = pd.Series([1, 2, 3], index=['p1', 'p2', 'p3'])
        variances = pd.Series([.6, .7, .8], index=['p1', 'p2', 'p3'])

        fs.update_after_step(volts, params, variances)

        expected = pd.Series([1, 2, 3, 4.4], index=start.index)

        pd.testing.assert_series_equal(expected, fs.suggest_next_position())


class NewtonSolverTest(unittest.TestCase):
    def test_position_suggestion(self):
        target = make_target(desired=pd.Series(index=["value1", "value2", "value3"], data=np.random.rand(3)))
        start = pd.Series(index=["position1", "position2", "position3"], data=np.random.rand(3))
        current_values = pd.Series(index=["value1", "value2", "value3"], data=np.random.rand(3))
        initial_gradient = np.random.rand(3, 3)
        gradient_estimators = []
        for i in range(3):
            gradient_estimators.append(FiniteDifferencesGradientEstimator(current_position=start,
                                                                          epsilon=1.,
                                                                          symmetric=True,
                                                                          current_estimate=pd.Series(
                                                                              initial_gradient[i][:])))

        solver = NewtonSolver(target=target, gradient_estimators=gradient_estimators, current_position=start,
                              current_values=current_values)

        next_step = solver.suggest_next_position()
        assert (np.allclose(a=target.desired.values,
                            b=current_values + np.asarray(solver.jacobian).dot(np.asarray(next_step - start))))

        # with fixed numbers

        target = make_target(desired=pd.Series(index=["value1", "value2", "value3"], data=[1.5, 2.5, 6]))
        start = pd.Series(index=["position1", "position2", "position3"], data=[1, 1, 1])
        current_values = pd.Series(index=["value1", "value2", "value3"], data=[1, 3, 0])
        # required diff = [.5, -.5, 6]
        # required diff position = [-.5, .5, 6]
        initial_gradient = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        gradient_estimators = []
        for i in range(3):
            gradient_estimators.append(FiniteDifferencesGradientEstimator(current_position=start,
                                                                          epsilon=1.,
                                                                          symmetric=True,
                                                                          current_estimate=pd.Series(
                                                                              initial_gradient[i][:])))

        solver = NewtonSolver(target=target, gradient_estimators=gradient_estimators, current_position=start,
                              current_values=current_values)

        next_step = solver.suggest_next_position()
        assert (np.allclose(a=target.desired.values,
                            b=current_values + np.asarray(solver.jacobian).dot(np.asarray(next_step - start))))
        self.assertAlmostEqual(next_step.tolist(), [.5, 1.5, 7])


