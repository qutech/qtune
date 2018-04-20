import unittest

import pandas as pd

from qtune.solver import ForwardingSolver


class ForwardingSolverTests(unittest.TestCase):
    def test_forwarding(self):
        p2v = pd.Series(['v1', 'v2', 'v3'], index=['p1', 'p2', 'p3'])

        start = pd.Series([10, 11, 12, 13], index=['v1', 'v2', 'v3', 'v4'])

        fs = ForwardingSolver(parameter_to_voltage=p2v, current_position=start)

        volts = pd.Series([1.1, 2.2, 3.3, 4.4], index=start.index)
        params = pd.Series([1, 2, 3], index=['p1', 'p2', 'p3'])
        variances = pd.Series([.6, .7, .8], index=['p1', 'p2', 'p3'])

        fs.update_after_step(volts, params, variances)

        expected = pd.Series([1, 2, 3, 4.4], index=start.index)

        pd.testing.assert_series_equal(expected, fs.suggest_next_voltage())
