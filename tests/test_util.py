import unittest

import qtune.util


class UtilTests(unittest.TestCase):
    def test_nth(self):

        test_list = [[], [], [], [], []]
        self.assertIs(test_list[3], qtune.util.nth(test_list, 3))
