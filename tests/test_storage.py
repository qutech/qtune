import unittest
import tempfile
import os

import numpy as np
import pandas as pd

from qtune.storage import HDF5Serializable, to_hdf5, from_hdf5


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
        self.temp_file.close()

    def tearDown(self):
        os.remove(self.temp_file.name)

    def test_pure_python_serialization(self):
        data = {'asd': [1, 2, 3],
                'ggg': [{'a': 1}, [1, 2, 3], (1, 2, 3), 'test_stringüöäß', None]}

        to_hdf5(self.temp_file.name, 'data', data)

        recovered_data = from_hdf5(self.temp_file.name, reserved=[])

        self.assertEqual({'data': data}, recovered_data)

    def test_referenced_object_serialization(self):
        l_1 = [8, 7, 6]
        data = {'asd': [1, 2, 3],
                'ggg': [{'a': 1}, [1, 2, 3], l_1],
                '9000': l_1}

        to_hdf5(self.temp_file.name, 'data', data)

        recovered_data = from_hdf5(self.temp_file.name, reserved=[])

        self.assertEqual({'data': data}, recovered_data)

        self.assertIs(recovered_data['data']['9000'], recovered_data['data']['ggg'][2])

    def test_numpy_serialization(self):
        data = {'asd': [1, 2, 3],
                'ggg': [{'a': 1}, [1, 2, 3], np.arange(2, 9)]}

        to_hdf5(self.temp_file.name, 'data', data)

        recovered_data = from_hdf5(self.temp_file.name, reserved=[])

        np.testing.assert_equal({'data': data}, recovered_data)

    def test_data_frame_serialization(self):
        df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], index=['asdf', 'b'], columns=[0, 1, 2])
        data = {'asd': [1, 2, 3],
                'ggg': [{'a': 1}, [1, 2, 3], df]}

        to_hdf5(self.temp_file.name, 'data', data)

        recovered_data = from_hdf5(self.temp_file.name, reserved=[])

        pd.testing.assert_frame_equal(recovered_data['data']['ggg'][2], data['ggg'][2])

        recovered_data['data']['ggg'].pop(2)
        data['ggg'].pop(2)
        self.assertEqual({'data': data}, recovered_data)

    def test_series_serialization(self):
        ser = pd.Series(data=[1., 53, 95.4], index=['asdf', 'b', 'hh'])
        data = {'asd': [1, 2, 3],
                'ggg': [{'a': 1}, [1, 2, 3], ser]}

        to_hdf5(self.temp_file.name, 'data', data)

        recovered_data = from_hdf5(self.temp_file.name, reserved=[])

        pd.testing.assert_series_equal(recovered_data['data']['ggg'][2], data['ggg'][2])

        recovered_data['data']['ggg'].pop(2)
        data['ggg'].pop(2)
        self.assertEqual({'data': data}, recovered_data)

    def test_custom_class(self):
        with self.assertRaises(AttributeError):
            class MyClass(metaclass=HDF5Serializable):
                def __init__(self, hallo, gib, mir, argumente=None):
                    self.args = dict(hallo=hallo, gib=gib, mir=mir, argumente=argumente)

        class MyClass(metaclass=HDF5Serializable):
            def __init__(self, hallo, gib, mir, argumente=None):
                self.args = dict(hallo=hallo, gib=gib, mir=mir, argumente=argumente)

            def to_hdf5(self):
                return self.args.copy()

            def __eq__(self, other):
                try:
                    np.testing.assert_equal(self.args, other.args)
                    return True
                except AssertionError:
                    return False

        args = dict(hallo=1, gib=[1, 2], mir={'a': 9}, argumente=np.arange(6, 8))
        obj = MyClass(**args)
        data = {'asd': [1, 2, 3],
                'ggg': [{'a': 1}, [1, 2, 3], obj]}

        to_hdf5(self.temp_file.name, 'data', data)

        recovered_data = from_hdf5(self.temp_file.name, reserved=[])

        np.testing.assert_equal({'data': data}, recovered_data)