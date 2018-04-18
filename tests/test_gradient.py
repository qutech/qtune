import unittest
import tempfile
import os

import numpy as np
import pandas as pd

from qtune.storage import serializables, to_hdf5, from_hdf5
from qtune.gradient import KalmanGradientEstimator
from qtune.kalman_gradient import KalmanGradient


class GradientSerializationTest(unittest.TestCase):
    def test_serialization(self):
        self.assertIn('KalmanGradientEstimator', serializables)

        kal_args = dict(n_params=3, n_gates=2,
                        state_transition_function=np.random.randn(6, 6),
                        initial_gradient=np.random.rand(3, 2),
                        initial_covariance_matrix=np.random.rand(6, 6),
                        measurement_covariance_matrix=np.random.rand(3, 3),
                        process_noise=np.random.rand(6, 6),
                        alpha=1.1)
        kal_grad = KalmanGradient(**kal_args)

        np.testing.assert_equal(kal_args, kal_grad.to_hdf5())

        est_args = dict(kalman_gradient=kal_grad, current_position=pd.Series([1, 2], index=['abba', 'xy']),
                        current_value=10.,
                        maximum_covariance=0.5)

        estimator = KalmanGradientEstimator(**est_args)
        hdf5 = estimator.to_hdf5()

        self.assertIs(hdf5['kalman_gradient'], kal_grad)
        pd.testing.assert_series_equal(hdf5['current_position'], est_args['current_position'])
        self.assertEqual(hdf5['current_value'], est_args['current_value'])
        self.assertEqual(hdf5['maximum_covariance'], est_args['maximum_covariance'])

        temp_file = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
        temp_file.close()

        try:
            to_hdf5(temp_file.name, 'estimator', estimator)

            recovered = from_hdf5(temp_file.name)['estimator']

            hdf5 = recovered.to_hdf5()
            pd.testing.assert_series_equal(hdf5['current_position'], est_args['current_position'])
            self.assertEqual(hdf5['current_value'], est_args['current_value'])
            self.assertEqual(hdf5['maximum_covariance'], est_args['maximum_covariance'])

            np.testing.assert_equal(kal_args, hdf5['kalman_gradient'].to_hdf5())
        finally:
            os.remove(temp_file.name)

