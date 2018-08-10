import unittest
import tempfile
import os
import random

import numpy as np
import pandas as pd
import sympy as sp

from qtune.storage import serializables, to_hdf5, from_hdf5
from qtune.gradient import KalmanGradientEstimator
from qtune.gradient import FiniteDifferencesGradientEstimator
from qtune.kalman_gradient import KalmanGradient


class GradientSerializationTest(unittest.TestCase):
    def test_serialization(self):
        self.assertIn('KalmanGradientEstimator', serializables)

        kal_args = dict(n_values=3, n_pos_dim=2,
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
                        maximum_covariance=0.5,
                        epsilon=.1)

        estimator = KalmanGradientEstimator(**est_args)
        hdf5 = estimator.to_hdf5()

        self.assertIs(hdf5['kalman_gradient'], kal_grad)
        pd.testing.assert_series_equal(hdf5['current_position'], est_args['current_position'])
        self.assertEqual(hdf5['current_value'], est_args['current_value'])
        if isinstance(est_args['maximum_covariance'], float):
            pd.testing.assert_series_equal(hdf5['maximum_covariance'],
                                           pd.Series(index=est_args['current_position'].index,
                                                     data=est_args['maximum_covariance']))
        else:
            pd.testing.assert_series_equal(hdf5['maximum_covariance'], est_args['maximum_covariance'])

        temp_file = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
        temp_file.close()

        try:
            to_hdf5(temp_file.name, 'estimator', estimator)

            recovered = from_hdf5(temp_file.name, reserved=[])['estimator']

            hdf5 = recovered.to_hdf5()
            pd.testing.assert_series_equal(hdf5['current_position'], est_args['current_position'])
            self.assertEqual(hdf5['current_value'], est_args['current_value'])
            if isinstance(est_args['maximum_covariance'], float):
                pd.testing.assert_series_equal(hdf5['maximum_covariance'],
                                               pd.Series(index=est_args['current_position'].index,
                                                         data=est_args['maximum_covariance']))
            else:
                pd.testing.assert_series_equal(hdf5['maximum_covariance'], est_args['maximum_covariance'])

            np.testing.assert_equal(kal_args, hdf5['kalman_gradient'].to_hdf5())
        finally:
            os.remove(temp_file.name)


class FiniteDiffGradientTest(unittest.TestCase):
    def test_require_update_measurement(self):

        for symmetric in [True, False]:
            f_d_args = dict(current_position=pd.Series(data=[1, 2, 3, 4], index=["a", "b", "c", 'd']),
                            epsilon=.1,
                            symmetric=symmetric)

            f_d_grad_estimator = FiniteDifferencesGradientEstimator(**f_d_args)
            if symmetric:
                n_meas = 2 * f_d_args['current_position'].size - 2
            else:
                n_meas = f_d_args['current_position'].size - 1

            requested_measurements = []

            if not symmetric:
                f_d_grad_estimator.update(f_d_args["current_position"], random.random(), 1,
                                          is_new_position=True)
                requested_measurements.append(f_d_args["current_position"].loc[["a", "b", "c"]])
            for i in range(n_meas):
                requested_measurements.append(f_d_grad_estimator.require_measurement(gates=["a", "b", "c"]))
                pd.testing.assert_index_equal(requested_measurements[-1].index, pd.Index(["a", "b", "c"]))
                f_d_grad_estimator.update(requested_measurements[-1], random.random(), 1, is_new_position=True)

            if symmetric:
                voltage_diff = (np.stack(requested_measurements[1::2]) - np.stack(requested_measurements[::2])).T
            else:
                voltage_diff = (np.stack(requested_measurements[1:]) - np.asarray(requested_measurements[0])).T

            for diff_vec in voltage_diff.T:
                if symmetric:
                    self.assertAlmostEqual(2 * f_d_args['epsilon'], np.linalg.norm(diff_vec))
                else:
                    self.assertAlmostEqual(f_d_args['epsilon'], np.linalg.norm(diff_vec))

            # check the surjectivity
            voltage_diff = sp.Matrix(voltage_diff)
            kern = voltage_diff.nullspace()
            self.assert_(len(kern) == 0)


class KalmanGradientTest(unittest.TestCase):
    def test_require_measurement(self):
        kalman_grad = KalmanGradient(n_pos_dim=3, n_values=1, initial_gradient=None,
                                     initial_covariance_matrix=np.diag([2, .9, .9]))
        kalman_args = dict(kalman_gradient=kalman_grad,
                           current_position=pd.Series(data=[1, 2, 3], index=["a", "b", "c"]), current_value=1.,
                           maximum_covariance=1., epsilon=0.1)

        kalman_grad_est = KalmanGradientEstimator(**kalman_args)

        req_meas = kalman_grad_est.require_measurement()
        self.assertAlmostEqual(0, np.linalg.norm(req_meas - kalman_args["current_position"] - [.1, 0, 0]))

        req_meas = kalman_grad_est.require_measurement(gates=["b", "a"])
        pd.testing.assert_series_equal(req_meas, kalman_args["current_position"].add(pd.Series(index=["a", "b", "c"],
                                                                                               data=[.1, 0, 0])))

        req_meas = kalman_grad_est.require_measurement(gates=["b", "c"])
        self.assertTrue(req_meas is None)

        req_meas = kalman_grad_est.require_measurement()
        self.assertAlmostEqual(0, np.linalg.norm(req_meas - kalman_args["current_position"] - [.1, 0, 0]))

        kalman_grad_est.update(position=req_meas, value=1, covariance=.01, is_new_position=True)
        req_meas = kalman_grad_est.require_measurement()
        self.assertTrue(req_meas is None)

        # test with tuned jacobian
        kalman_grad = KalmanGradient(n_pos_dim=3, n_values=1, initial_gradient=None,
                                     initial_covariance_matrix=np.diag([2, .9, .9]))
        kalman_args = dict(kalman_gradient=kalman_grad,
                           current_position=pd.Series(data=[1, 2, 3], index=["a", "b", "c"]), current_value=1.,
                           maximum_covariance=1., epsilon=0.1)

        kalman_grad_est = KalmanGradientEstimator(**kalman_args)

        tuned_jacobian = pd.DataFrame(data=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]), index=["a", "b", "c"],
                                      columns=["a", "b", "c"])

        req_meas = kalman_grad_est.require_measurement(tuned_jacobian=tuned_jacobian)
        expected_diff = pd.Series(index=['a', 'b', 'c'], data=[-1 * kalman_args['epsilon'] / np.sqrt(2),
                                                               kalman_args['epsilon'] / np.sqrt(2), 0])
        pd.testing.assert_series_equal(req_meas, kalman_args['current_position'].add(expected_diff))
