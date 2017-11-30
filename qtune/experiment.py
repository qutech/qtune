from typing import Tuple, Dict, Any


import pandas

__all__ = ['Experiment', 'Measurement', 'GateIdentifier']

GateIdentifier = str


class Measurement(str):
    def __new__(cls, name, **kwargs):
        return super().__new__(cls, name)

    def __init__(self, name, **kwargs):
        super().__init__()

        self.parameter = kwargs


class Experiment:
    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        raise NotImplementedError()

    @property
    def gate_voltages(self) -> Tuple[GateIdentifier, ...]:
        raise NotImplementedError()

    def measure(self,
                gate_voltages: pandas.Series,
                measurement: Measurement) -> pandas.Series:
        """Conduct specified measurements with given gate_voltages

        :param gate_voltages:
        :param measurement:
        :return:
        """
        raise NotImplementedError()
