import unittest
import unittest.mock
import pandas.testing

import pandas as pd
import numpy as np

from typing import List

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

    def test_update_after_step(self):
        target = make_target(desired=pd.Series(index=["value1", "value2"], data=np.random.rand(2)))
        start = pd.Series(index=["position1", "position2", "position3"], data=np.random.rand(3))
        current_values = pd.Series(index=["value1", "value2", "value3"], data=np.random.rand(3))
        initial_gradient = np.random.rand(2, 3)
        gradient_estimators = []
        for i in range(2):
            gradient_estimators.append(FiniteDifferencesGradientEstimator(current_position=start,
                                                                          epsilon=1.,
                                                                          symmetric=True,
                                                                          current_estimate=pd.Series(
                                                                              initial_gradient[i][:])))
            gradient_estimators[-1].update = unittest.mock.Mock()
        solver = NewtonSolver(target=target, gradient_estimators=gradient_estimators,
                              current_position=start,
                              current_values=current_values)

        update_args = dict(position=pd.Series(index=["position2", "position3", "position1"], data=[2, 3, 1]),
                           values=pd.Series(index=["value3", "value2", "value1", "value4"], data=[3, 2, 1, 4]),
                           variances=pd.Series(index=["value1", "value2", "value4", "value3"], data=[2, 4, 8, 6]))
        solver.update_after_step(**update_args)

        # assert_called_with cannot be used because the overloaded == operator doesnt return bool values for pd.Series
        self.assertEqual(gradient_estimators[0].update.call_count, 1)
        call_args_1 = gradient_estimators[0].update.call_args
        pandas.testing.assert_series_equal(call_args_1[0][0], update_args["position"][start.index])
        self.assertAlmostEqual(call_args_1[0][1:],
                               (update_args["values"][target.index[0]], update_args["variances"][target.index[0]]))
        self.assertEqual(call_args_1[1], dict(is_new_position=True))

        self.assertEqual(gradient_estimators[0].update.call_count, 1)
        call_args_2 = gradient_estimators[1].update.call_args
        pandas.testing.assert_index_equal(call_args_2[0][0].index, pd.Index(["position1", "position2", "position3"]))
        self.assertAlmostEqual(call_args_2[0][1:], (2, 4))
        self.assertEqual(call_args_2[1], dict(is_new_position=True))

