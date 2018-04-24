import warnings
from typing import Union

import h5py
import numpy as np
import pandas as pd


__all__ = ["serializables", "HDF5Serializable"]


serializables = dict()


def _get_dtype(arr):
    if arr.dtype == 'O':
        if all(isinstance(t, str) for t in arr.ravel()):
            return h5py.special_dtype(vlen=str)
    return arr.dtype


class HDF5Serializable(type):
    def __new__(mcs, name, bases, attrs):
        if name in serializables:
            warnings.warn("Overwriting known serializable {}".format(name))
        cls = type.__new__(mcs, name, bases, attrs)

        serializables[name] = cls
        return cls

    def __init__(cls, name, bases, attrs):
        type.__init__(cls, name, bases, attrs)
        if 'to_hdf5' not in cls.__dict__:
            raise AttributeError('Missing method: "to_hdf5" that should return all constructor arguments', name)


def _to_hdf5(hdf5_parent_group: h5py.Group, name, obj, serialized):
    if id(obj) in serialized:
        hdf5_parent_group.create_dataset(name, data=serialized[id(obj)].ref)
        return

    if isinstance(obj, dict):
        hdf5_group = hdf5_parent_group.create_group(name)
        hdf5_group.attrs['#type'] = 'dict'
        for key, value in obj.items():
            _to_hdf5(hdf5_group, key, value, serialized)
        serialized[id(obj)] = hdf5_group
        return

    if isinstance(obj, (list, tuple)):
        hdf5_group = hdf5_parent_group.create_group(name)
        hdf5_group.attrs['#type'] = 'list' if isinstance(obj, list) else 'tuple'
        for idx, value in enumerate(obj):
            _to_hdf5(hdf5_group, str(idx), value, serialized)
        serialized[id(obj)] = hdf5_group
        return

    if type(obj).__name__ in serializables:
        hdf5_group = hdf5_parent_group.create_group(name)
        hdf5_group.attrs['#type'] = type(obj).__name__

        serialized[id(obj)] = hdf5_group

        data = obj.to_hdf5()
        for key, value in data.items():
            _to_hdf5(hdf5_group, key, value, serialized)
        return

    if isinstance(obj, pd.DataFrame):
        dset = hdf5_parent_group.create_dataset(name, data=obj)
        dset.attrs.create('index', data=obj.index, dtype=_get_dtype(obj.index))
        dset.attrs.create('columns', data=obj.columns, dtype=_get_dtype(obj.columns))
        dset.attrs['#type'] = 'DataFrame'

        serialized[id(obj)] = dset
        return

    if isinstance(obj, pd.Series):
        dset = hdf5_parent_group.create_dataset(name, data=obj)
        dset.attrs.create('index', data=obj.index, dtype=_get_dtype(obj.index))
        dset.attrs['#type'] = 'Series'

        serialized[id(obj)] = dset
        return

    if isinstance(obj, np.ndarray):
        dset = hdf5_parent_group.create_dataset(name, data=obj)
        serialized[id(obj)] = dset
        return

    if isinstance(obj, (float, int, complex)):
        hdf5_parent_group.create_dataset(name, data=obj, shape=())
        return

    if isinstance(obj, str):
        dt = h5py.special_dtype(vlen=str)
        dset = hdf5_parent_group.create_dataset(name, data=obj, dtype=dt)
        dset.attrs['#type'] = 'str'
        serialized[id(obj)] = dset
        return

    if obj is None:
        dset = hdf5_parent_group.create_dataset(name, dtype="f")
        dset.attrs['#type'] = 'NoneType'
        serialized[id(obj)] = dset
        return

    raise RuntimeError()


def to_hdf5(filename_or_handle: Union[str, h5py.Group], name: str, obj):
    if isinstance(filename_or_handle, h5py.Group):
        root = filename_or_handle
    else:
        root = h5py.File(filename_or_handle, mode='r+')

    _to_hdf5(root, name, obj, dict())


def _from_hdf5(root: h5py.File, hdf5_obj: h5py.HLObject, deserialized=None):
    if isinstance(hdf5_obj, h5py.Reference):
        hdf5_obj = root[hdf5_obj]

    if hdf5_obj.id in deserialized:
        return deserialized[hdf5_obj.id]

    if isinstance(hdf5_obj, h5py.Group):
        if '#type' not in hdf5_obj.attrs or hdf5_obj.attrs['#type'] == 'dict':
            deserialized[hdf5_obj.id] = dict()
            result = deserialized[hdf5_obj.id]

            for k in hdf5_obj.keys():
                result[k] = _from_hdf5(root, hdf5_obj[k], deserialized)
            return result

        elif hdf5_obj.attrs['#type'] in serializables:
            cls = serializables[hdf5_obj.attrs['#type']]
            deserialized[hdf5_obj.id] = cls(
                **{k: _from_hdf5(root, v, deserialized)
                   for k, v in hdf5_obj.items()}
            )
            return deserialized[hdf5_obj.id]

        elif hdf5_obj.attrs['#type'] == 'list':
            deserialized[hdf5_obj.id] = list()
            result = deserialized[hdf5_obj.id]

            for idx in range(len(hdf5_obj.keys())):
                result.append(_from_hdf5(root, hdf5_obj[str(idx)], deserialized))
            return result

        elif hdf5_obj.attrs['#type'] == 'tuple':
            result = []
            for idx in range(len(hdf5_obj.keys())):
                result.append(_from_hdf5(root, hdf5_obj[str(idx)], deserialized))
            result = tuple(result)
            deserialized[hdf5_obj.id] = result
            return result


        else:
            warnings.warn('Unknown type: ', hdf5_obj.attrs['#type'])

    elif isinstance(hdf5_obj, h5py.Dataset):
        if '#type' in hdf5_obj.attrs:
            if hdf5_obj.attrs['#type'] == 'DataFrame':
                idx = hdf5_obj.attrs['index']
                col = hdf5_obj.attrs['columns']
                result = pd.DataFrame(data=np.asarray(hdf5_obj), index=idx, columns=col)
                deserialized[hdf5_obj.id] = result
                return result

            if hdf5_obj.attrs['#type'] == 'Series':
                idx = hdf5_obj.attrs['index']
                result = pd.Series(data=np.asarray(hdf5_obj), index=idx)
                deserialized[hdf5_obj.id] = result
                return result

            if hdf5_obj.attrs['#type'] == 'str':
                result = str(np.asarray(hdf5_obj))
                deserialized[hdf5_obj.id] = result
                return result

            if hdf5_obj.attrs['#type'] == 'NoneType':
                return None

        result = np.asarray(hdf5_obj)
        if result.shape == ():
            result = result[()]
        if isinstance(result, h5py.Reference):
            return _from_hdf5(root, result, deserialized)
        else:
            return result

    else:
        raise RuntimeError()


def from_hdf5(filename_or_handle):
    if isinstance(filename_or_handle, h5py.Group):
        root = filename_or_handle
    else:
        root = h5py.File(filename_or_handle, mode='r')

    return _from_hdf5(root, root, dict())
