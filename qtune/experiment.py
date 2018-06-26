from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

from qtune.util import time_string
from qtune.storage import HDF5Serializable

__all__ = ['Experiment', 'Measurement', 'GateIdentifier']

GateIdentifier = str


class Measurement(metaclass=HDF5Serializable):
    """
    This class saves all necessary information for a measurement.
    """
    def __init__(self, name, **kwargs):
        super().__init__()
        self._name = name
        self.options = kwargs

    @property
    def name(self):
        return self._name

    def get_file_name(self):
        return time_string()

    def to_hdf5(self):
        return dict(self.options,
                    name=self.name)

    def __repr__(self):
        return "{type}({data})".format(type=type(self), data=self.to_hdf5())


class Experiment:
    """
    Basic class implementing the structure of an experiment consisting of gates whose voltages can be set and read.
    Additionally we require the possibility to conduct measurements.
    """
    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        """Return available measurements for the Experiment"""
        raise NotImplementedError()

    @property
    def gate_voltage_names(self) -> Tuple[str]:
        raise NotImplementedError()

    def read_gate_voltages(self) -> pd.Series:
        raise NotImplementedError()

    def set_gate_voltages(self, new_gate_voltages: pd.Series) -> pd.Series:
        """
        Set the gate Voltages and return the voltages which have actually been set. (i.e. when the Voltages are saved
        in a different format.)
        :param new_gate_voltages:
        :return: actually set voltages.
        """
        raise NotImplementedError()

    def measure(self, measurement: Measurement) -> np.ndarray:
        """Conduct specified measurement

        :param measurement:
        :return data:
        """
        raise NotImplementedError()
