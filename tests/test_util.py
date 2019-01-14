import unittest
import numpy as np

import qtune.util


class UtilTests(unittest.TestCase):
    def test_nth(self):

        test_list = [[], [], [], [], []]
        self.assertIs(test_list[3], qtune.util.nth(test_list, 3))

    def test_gradient_min_evaluations_asymmetric(self):
        gradient = np.array([[1., 2., 3],
                             [4., 0, 1.]])

        vec0 = np.asarray([1., 2., 3.])
        dv1 = np.asarray([.007, .007, 0.])
        dv2 = np.asarray([.005, 0., 0.])
        dv3 = np.asarray([0., .002, .005])

        deltas = [np.asarray([0., 0., 0.]), dv1, dv2, dv3]
        voltage_points = [vec0 + dv for dv in deltas]

        p0 = np.array([10., 11.])
        parameters = [p0 + gradient @ dv for dv in deltas]

        grad = qtune.util.gradient_min_evaluations(parameters=parameters, voltage_points=voltage_points)

        np.testing.assert_almost_equal(grad, gradient)

    def test_gradient_min_evaluations_symmetric(self):
        gradient = np.array([[1., 2., 3],
                             [4., 0, 1.]])

        vec0 = np.asarray([1., 2., 3.])
        dv1 = np.asarray([.007, .007, 0.])
        dv2 = np.asarray([.005, 0., 0.])
        dv3 = np.asarray([0., .002, .005])

        deltas = [dv1, -dv1, dv2, -dv2, -dv3, dv3]
        voltage_points = [vec0 + dv for dv in deltas]

        p0 = np.array([10., 11.])
        parameters = [p0 + gradient @ dv for dv in deltas]

        grad = qtune.util.gradient_min_evaluations(parameters=parameters, voltage_points=voltage_points)

        np.testing.assert_almost_equal(grad, gradient)

    def test_gradient_min_evaluation_exception(self):
        gradient = np.array([[1., 2., 3],
                             [4., 0, 1.]])

        vec0 = np.asarray([1., 2., 3.])
        dv1 = np.asarray([.007, .007, 0.])
        dv2 = np.asarray([.005, 0., 0.])
        dv3 = dv1 + dv2

        deltas = [dv1, -dv1, dv2, -dv2, -dv3, dv3]
        voltage_points = [vec0 + dv for dv in deltas]

        p0 = np.array([10., 11.])
        parameters = [p0 + gradient @ dv for dv in deltas]

        with self.assertRaises(RuntimeError):
            qtune.util.gradient_min_evaluations(parameters=parameters[:-1], voltage_points=voltage_points[:-1])

        with self.assertRaises(qtune.util.EvaluationError):
            qtune.util.gradient_min_evaluations(parameters=parameters, voltage_points=voltage_points)

    def test_calculate_gradient_non_orthogonal_symmetric(self):
        gradient = np.array([1., 2., 3])

        vec0 = np.asarray([1., 2., 3.])
        dv1 = np.asarray([0.01, .01, 0.])
        dv2 = np.asarray([1, 0., 0.])
        dv3 = np.asarray([0., .0, 1])

        deltas = [dv1, -dv1, dv2, -(dv2 + dv1), -dv3, dv3]
        voltage_points = [vec0 + dv for dv in deltas]

        variances = [1, 10, 100, 1000, 10000, 100000]

        parameters = [-10 + gradient @ dv for dv in deltas]

        grad, var = qtune.util.calculate_gradient_non_orthogonal(positions=voltage_points,
                                                                 values=parameters,
                                                                 variances=variances)

        np.testing.assert_almost_equal(grad, gradient)

    def test_calculate_gradient_non_orthogonal_asymmetric(self):
        gradient = np.array([[1., 2., 3]])

        vec0 = np.asarray([1., 2., 3.])
        dv1 = np.asarray([0.01, .01, 0.])
        dv2 = np.asarray([1, 0., 0.])
        dv3 = np.asarray([0., .0, 1])

        deltas = [np.asarray([0., 0., 0.]), dv1, dv2, dv3]
        voltage_points = [vec0 + dv for dv in deltas]

        parameters = [-10 + gradient @ dv for dv in deltas]
        variances = [1e-6, 10e-6, 100e-6, 10000e-6]

        grad, var = qtune.util.calculate_gradient_non_orthogonal(positions=voltage_points,
                                                                 values=parameters,
                                                                 variances=variances)

        np.testing.assert_almost_equal(grad, gradient)
