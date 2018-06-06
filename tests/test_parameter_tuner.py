import unittest
import unittest.mock
import pandas.testing
from unittest.mock import patch
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from qtune.parameter_tuner import SubsetTuner
from qtune.solver import make_target


class SubsetTunerTest(unittest.TestCase):
    @patch("qtune.solver.Solver")
    @patch("qtune.evaluator.Evaluator") # use side_effect to return different mocks each time
    def test_evaluate_is_tuned(self, mocked_evaluator, mocked_solver):
        evaluators = []
        target = make_target(desired=pd.Series(index=["parameter_0", "parameter_1"], data=np.random.rand(2)))
        for i in range(2):
            evaluator = mocked_evaluator()
            evaluator.evaluate = MagicMock(return_value=(pd.Series(index=["parameter_" + str(i)], data=10 * i),
                                                         pd.Series(index=["parameter_" + str(i)], data=i)))
            evaluator.parameters = ("parameter_" + str(i), )
            evaluators.append(evaluator)

        solver = mocked_solver()
        solver.target = target
        s_t_tuner_args = dict(evaluators=evaluators,
                              gates=["a", "b", "c"],
                              solver=solver)

        subset_tuner = SubsetTuner(**s_t_tuner_args)
        voltages = pd.Series(index=s_t_tuner_args['gates'])
        subset_tuner.is_tuned(voltages=voltages)


