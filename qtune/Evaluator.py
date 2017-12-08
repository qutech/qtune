from qtune.experiment import Experiment, Measurement
from typing import Tuple
import pandas as pd


class Evaluator:
    def __init__(self, experiment: Experiment, measurements: Tuple[Measurement, ...], parameters: pd.Series):
        self.experiment = experiment
        self.measurements = measurements
        self.parameters = parameters

    def __call__(self, *args, **kwargs):
        return self.evaluate()

    def evaluate(self) -> pd.Series:
        raise NotImplementedError
