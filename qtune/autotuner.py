import h5py
from qtune.util import time_string
from qtune.experiment import Experiment
from typing import List
from qtune.parameter_tuner import ParameterTuner
from qtune.storage import to_hdf5, HDF5Serializable
import qtune.util


class Autotuner(metaclass=HDF5Serializable):
    """
    The auto tuner class combines the evaluator and solver classes to tune an experiment.
    """

    def __init__(self, experiment: Experiment, tuning_hierarchy: List[ParameterTuner] = None, current_tuner_index=0,
                 current_tuner_status=False, voltage_to_set=None, hdf5_filename=None):
        self._experiment = experiment
        self._tuning_hierarchy = tuning_hierarchy
        self._current_tuner_index = current_tuner_index
        self._current_tuner_status = current_tuner_status
        self._voltage_to_set = voltage_to_set
        self._hdf5_filename = hdf5_filename

    def tuning_complete(self) -> bool:
        if self._current_tuner_index == len(self._tuning_hierarchy):
            return True
        else:
            return False

    def ready_to_tune(self) -> bool:
        raise NotImplementedError

    def get_current_tuner(self):
        return self._tuning_hierarchy[self._current_tuner_index]

    def iterate(self):
        if self._voltage_to_set is not None:
            self._experiment.set_gate_voltages(self._voltage_to_set)
            self._current_tuner_index = 0
            self._voltage_to_set = None
        elif self._current_tuner_status is False:
            if self.get_current_tuner().is_tuned(self._experiment.read_gate_voltages()):
                self._current_tuner_index += 1
            else:
                self._current_tuner_status = True
        else:
            self._voltage_to_set = self.get_current_tuner().get_next_voltages()
            self._current_tuner_status = False

    def autotune(self):
#        if not self.ready_to_tune():
#            print("Setup incomplete!")
#            return
        while not self.tuning_complete():
            self.iterate()
            if self._hdf5_filename:
                filename = self._hdf5_filename + r"\\" + time_string() + ".hdf5"
            else:
                filename = time_string() + ".hdf5"
            hdf5_file = h5py.File(filename, 'w-')
            to_hdf5(hdf5_file, name="autotuner", obj=self, reserved={"experiment": self._experiment})

    def to_hdf5(self):
        return dict(
            experiment=self._experiment,
            tuning_hierarchy=self._tuning_hierarchy,
            current_tuner_index=self._current_tuner_index,
            current_tuner_status=self._current_tuner_status,
            voltage_to_set=self._voltage_to_set,
            hdf5_filename=self._hdf5_filename
        )
