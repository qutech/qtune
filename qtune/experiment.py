from abc import ABCMeta, abstractmethod
from typing import Tuple, Collection, Dict, Any


import pandas

__all__ = ['Experiment', 'Measurement', 'GateIdentifier']

GateIdentifier = str


class Experiment(metaclass=ABCMeta):
    @abstractmethod
    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        pass

    @abstractmethod
    @property
    def gate_voltages(self) -> Tuple[GateIdentifier, ...]:
        pass

    @abstractmethod
    def measure(self,
                gate_voltages: pandas.Series,
                measurements: Collection[Measurement]) -> pandas.Series:
        """Conduct specified measurements with given gate_voltages

        :param gate_voltages:
        :param measurements:
        :return:
        """
        pass


class Measurement(str):
    def __new__(cls, name, **kwargs):
        return super().__new__(cls, name)

    def __init__(self, name, **kwargs):
        super().__init__(name)

        self.parameter = kwargs