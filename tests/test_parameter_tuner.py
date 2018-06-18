import unittest
import unittest.mock
from unittest.mock import patch
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from qtune.parameter_tuner import SubsetTuner, SensingDotTuner
from qtune.solver import make_target


class SubsetTunerTest(unittest.TestCase):
    @patch("qtune.solver.Solver")
    @patch("qtune.evaluator.Evaluator", side_effect=unittest.mock.Mock)
    def test_evaluate_is_tuned(self, mocked_evaluator, mocked_solver):
        evaluators = []
        n_evaluator = 2
        target = make_target(desired=pd.Series(index=["parameter_" + str(i) for i in range(n_evaluator)],
                                               data=[i + 1 for i in range(n_evaluator)]),
                             tolerance=pd.Series(index=["parameter_" + str(i) for i in range(n_evaluator)],
                                                 data=.56))
        for i in range(n_evaluator):
            evaluator = mocked_evaluator()
            evaluator.evaluate = MagicMock(return_value=(pd.Series(index=["parameter_" + str(i)], data=10 * i),
                                                         pd.Series(index=["parameter_" + str(i)], data=i)))
            evaluator.parameters = ("parameter_" + str(i), )
            evaluators.append(evaluator)

        solver = mocked_solver()
        solver.target = target
        subset_tuner_args = dict(evaluators=evaluators,
                                 gates=["a", "b", "c"],
                                 solver=solver)

        subset_tuner = SubsetTuner(**subset_tuner_args)
        solver_voltages = pd.Series(index=subset_tuner_args['gates'], data=[5, 10, 30.])
        full_voltages = solver_voltages.copy(deep=True)
        assert(isinstance(full_voltages, pd.Series))
        full_voltages["d"] = 1.
        failing_is_tuned = subset_tuner.is_tuned(voltages=full_voltages)

        # assert that the solver is called with the right arguments
        self.assertEqual(solver.update_after_step.call_count, 1)
        pd.testing.assert_series_equal(full_voltages, solver.update_after_step.call_args[0][0])

        parameter = pd.Series(data=[10 * i for i in range(n_evaluator)],
                              index=["parameter_" + str(i) for i in range(n_evaluator)])
        variances = pd.Series(data=[i for i in range(n_evaluator)],
                              index=["parameter_" + str(i) for i in range(n_evaluator)])
        pd.testing.assert_series_equal(solver.update_after_step.call_args[0][1], parameter)
        pd.testing.assert_series_equal(solver.update_after_step.call_args[0][2], variances)

        # assert that the subset tuner updates his attributes
        pd.testing.assert_series_equal(subset_tuner._last_voltage, full_voltages)
        pd.testing.assert_series_equal(subset_tuner.last_parameter_covariance[0], parameter)
        pd.testing.assert_series_equal(subset_tuner.last_parameter_covariance[1], variances)

        # assert that is tuned returns the correct state
        self.assertEqual(failing_is_tuned, False)

        subset_tuner.evaluate = unittest.mock.Mock(return_value=(target.desired, variances))

        passing_is_tuned = subset_tuner.is_tuned(voltages=full_voltages)

        self.assertEqual(passing_is_tuned, True)


class SensingDotTunerTest(unittest.TestCase):
    @patch("qtune.solver.ForwardingSolver")
    @patch("qtune.evaluator.Evaluator", side_effect=unittest.mock.Mock)
    def test_evaluate_is_tuned(self, mocked_evaluator, mocked_solver):
        target = make_target(minimum=pd.Series(index=["position_a", "current_signal", "optimal_signal"],
                                               data=[np.nan, 1., 1.]),
                             cost_threshold=pd.Series(index=["position_a", "current_signal", "optimal_signal"],
                                                      data=[np.nan, np.nan, 1.]))
        first_return_cheap = (pd.Series(index=["position_a", "current_signal", "optimal_signal"],
                                        data=[.2, .8, .9]),
                              pd.Series(index=["position_a", "current_signal", "optimal_signal"],
                                        data=.1))
        # gate a starts at .1 with a signal of .8
        # the signal can be improved by going to .2 up to .9 (still not good enough)
        # the expensive evaluator finds a new position for gate b and a
        # the cheap evaluator brings gate a to .4 to get the final signal of 1.1
        second_return_cheap = (pd.Series(index=["position_a", "current_signal", "optimal_signal"],
                                         data=[.4, .1, 1.1]),
                               pd.Series(index=["position_a", "current_signal", "optimal_signal"],
                                         data=.1))
        return_expensive = (pd.Series(index=["position_a", "position_b"], data=.3),
                            pd.Series(index=["position_a", "position_b"], data=.01))
        cheap_evaluator = mocked_evaluator()
        cheap_evaluator.evaluate = MagicMock(side_effect=[first_return_cheap, second_return_cheap])
        expensive_evaluator = mocked_evaluator()
        expensive_evaluator.evaluate = MagicMock(return_value=return_expensive)
        cheap_evaluator.parameters = ["position_a", "current_signal", "optimal_signal"]
        expensive_evaluator.parameters = ["position_a", "position_b"]

        solver = mocked_solver()
        solver.target = target
        sensing_dot_tuner_args = dict(cheap_evaluators=(cheap_evaluator, ),
                                      expensive_evaluators=(expensive_evaluator, ),
                                      gates=["a", "b"],
                                      solver=solver)

        sensing_dot_tuner = SensingDotTuner(**sensing_dot_tuner_args)
        solver_voltages = pd.Series(index=sensing_dot_tuner_args['gates'], data=.1)
        full_voltages = solver_voltages.copy(deep=True)
        assert (isinstance(full_voltages, pd.Series))
        full_voltages["d"] = 1.
        failing_is_tuned = sensing_dot_tuner.is_tuned(voltages=full_voltages)

        # assert that the solver is called with the right arguments
        self.assertEqual(solver.update_after_step.call_count, 1)
        self.assertEqual(cheap_evaluator.evaluate.call_count, 1)
        self.assertEqual(expensive_evaluator.evaluate.call_count, 1)
        pd.testing.assert_series_equal(solver_voltages, solver.update_after_step.call_args[0][0])

        parameter = pd.Series(data=[.3, .3], index=["position_a", "position_b"])
        variances = pd.Series(data=[.01, .01], index=["position_a", "position_b"])
        pd.testing.assert_series_equal(solver.update_after_step.call_args[0][1], parameter)
        pd.testing.assert_series_equal(solver.update_after_step.call_args[0][2], variances)

        # assert that the subset tuner updates his attributes
        pd.testing.assert_series_equal(sensing_dot_tuner._last_voltage, full_voltages)
        updated_parameter = pd.Series(data=[.3, .3, .8, .9],
                                      index=["position_a", "position_b", "current_signal", "optimal_signal"])
        updated_variances = pd.Series(data=[.01, .01, .1, .1],
                                      index=["position_a", "position_b", "current_signal", "optimal_signal"])
        pd.testing.assert_series_equal(sensing_dot_tuner.last_parameter_covariance[0], updated_parameter.sort_index())
        pd.testing.assert_series_equal(sensing_dot_tuner.last_parameter_covariance[1], updated_variances.sort_index())

        # assert that is tuned returns the correct state
        self.assertEqual(failing_is_tuned, False)

        new_voltages = pd.Series(index=["a", "b", "d"], data=.3)
        passing_is_tuned = sensing_dot_tuner.is_tuned(voltages=new_voltages)

        self.assertEqual(passing_is_tuned, False)

        # assert that the solver is called with the right arguments at the second call
        self.assertEqual(solver.update_after_step.call_count, 2)
        self.assertEqual(cheap_evaluator.evaluate.call_count, 2)
        self.assertEqual(expensive_evaluator.evaluate.call_count, 1)
        pd.testing.assert_series_equal(new_voltages[pd.Index(["a", "b"])], solver.update_after_step.call_args[0][0])

        pd.testing.assert_series_equal(solver.update_after_step.call_args[0][1], second_return_cheap[0].sort_index())
        pd.testing.assert_series_equal(solver.update_after_step.call_args[0][2], second_return_cheap[1].sort_index())
