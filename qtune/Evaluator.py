from qtune.experiment import Experiment, Measurement, TestExperiment
from typing import Tuple
import pandas as pd
import numpy as np
import h5py


class Evaluator:
    def __init__(self, experiment: Experiment, measurements: Tuple[Measurement, ...], parameters: pd.Series):
        self.experiment = experiment
        self.measurements = measurements
        self.parameters = parameters

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
        raise NotImplementedError


class TestEvaluator(Evaluator):
    def __init__(self, experiment: TestExperiment, measurements=None,
                 parameters=pd.Series((np.nan, np.nan), ('linsine', 'quadratic')), ):
        super().__init__(experiment=experiment, measurements=measurements, parameters=parameters)

    def evaluate(self, storing_group: h5py.Group) -> pd.Series:
        test_voltages = self.experiment.read_gate_voltages()
        test_voltages = test_voltages.sort_index()
        linsine = test_voltages[0] * np.sin(test_voltages[1])
        quadratic = test_voltages[1] * test_voltages[1]
        return pd.Series([linsine, quadratic, False], ['linsine', 'quadratic', 'failed'])
