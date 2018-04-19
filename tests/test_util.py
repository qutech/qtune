import unittest
import numpy as np

import qtune.util


class UtilTests(unittest.TestCase):
    def test_nth(self):

        test_list = [[], [], [], [], []]
        self.assertIs(test_list[3], qtune.util.nth(test_list, 3))

    def test_gradient_min_evaluations(self):

        vec0 = np.asarray([0., 0., 0.])
        vec1 = np.asarray([.007, .007, 0.])
        vec2 = np.asarray([.005, 0., 0.])
        vec3 = np.asarray([0., .002, .005])
        vec4 = np.asarray([2., 2., 2.])
        voltage_points = [vec0, vec1, vec2, vec3]
        parameters = [(a + vec4)**2 for a in voltage_points]

        grad = qtune.util.gradient_min_evaluations(parameters=parameters, voltage_points=voltage_points)

        self.assertLess(a=np.linalg.norm(grad - np.diag([4., 4., 4.])), b=.01)

        voltage_points = [-vec1, vec1, -vec2, vec2, -vec3, vec3]
        parameters = [(a + vec4)**2 for a in voltage_points]

        grad = qtune.util.gradient_min_evaluations(parameters=parameters, voltage_points=voltage_points)
        self.assertLess(a=np.linalg.norm(grad - np.diag([4., 4., 4.])), b=.0001)


