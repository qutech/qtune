import unittest
import unittest.mock
import pandas.testing
from unittest.mock import patch
from unittest.mock import MagicMock

import pandas as pd

from qtune.parameter_tuner import SubsetTuner
from qtune.solver import make_target


class SubsetTunerTest(unittest.TestCase):
    @patch("qtune.solver.Solver", side_effect=unittest.mock.Mock)
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
        voltages = pd.Series(index=subset_tuner_args['gates'], data=[5, 10, 30.])
        failing_is_tuned = subset_tuner.is_tuned(voltages=voltages)

        self.assertEqual(solver.update_after_step.call_count, 1)
        pd.testing.assert_series_equal(voltages, solver.update_after_step.call_args[0][0])

        # assert that the solver is called with the right arguments
        parameter = pd.Series(data=[10 * i for i in range(n_evaluator)],
                              index=["parameter_" + str(i) for i in range(n_evaluator)])
        variances = pd.Series(data=[i for i in range(n_evaluator)],
                              index=["parameter_" + str(i) for i in range(n_evaluator)])
        pd.testing.assert_series_equal(solver.update_after_step.call_args[0][1], parameter)
        pd.testing.assert_series_equal(solver.update_after_step.call_args[0][2], variances)

        # assert that the subset tuner updates his attributes
        pd.testing.assert_series_equal(subset_tuner._last_voltage, voltages)
        pd.testing.assert_series_equal(subset_tuner.last_parameter_covariance[0], parameter)
        pd.testing.assert_series_equal(subset_tuner.last_parameter_covariance[1], variances)

        self.assertEqual(failing_is_tuned, False)

        subset_tuner.evaluate = unittest.mock.Mock(return_value=(target.desired, variances))

        passing_is_tuned = subset_tuner.is_tuned(voltages=voltages)

        self.assertEqual(passing_is_tuned, True)






