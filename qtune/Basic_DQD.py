import pandas as pd
from typing import Tuple
from qtune.experiment import Experiment, Measurement


class BasicDQD(Experiment):
    default_line_scan = Measurement('line_scan',
                                    center=0., range=3e-3, gate='RFA', N_points=1280, ramptime=.0005,
                                    N_average=3, AWGorDecaDAC='DecaDAC')
    default_detune_scan = Measurement('detune_scan',
                                      center=0., range=2e-3, N_points=100, ramptime=.02,
                                      N_average=20, AWGorDecaDAC='DecaDAC')
    default_lead_scan = Measurement('lead_scan', gate='B', AWGorDecaDAC='DecaDAC')

    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        return self.default_line_scan, self.default_detune_scan, self.default_lead_scan

    def tune_qpc(self, qpc_position=None, tuning_range=3e-3):
        raise NotImplementedError()

    def read_qpc_voltage(self) -> pd.Series:
        raise NotImplementedError()
