import h5py
import os
import pandas as pd
from qtune.util import time_string
from qtune.experiment import Experiment
from typing import List, Optional
from qtune.parameter_tuner import ParameterTuner, SubsetTuner
from qtune.solver import NewtonSolver
from qtune.storage import to_hdf5, HDF5Serializable, from_hdf5


class Autotuner(metaclass=HDF5Serializable):
    """
    The auto tuner class combines the evaluator and solver classes to tune an experiment.
    """

    def __init__(self, experiment: Experiment, tuning_hierarchy: List[ParameterTuner] = None,
                 current_tuner_index: int = 0, current_tuner_status: bool = False,
                 voltage_to_set: Optional[pd.Series] = None, hdf5_storage_path: str = None):
        self._experiment = experiment
        self._tuning_hierarchy = tuning_hierarchy
        self._current_tuner_index = current_tuner_index
        self._current_tuner_status = current_tuner_status
        self._voltage_to_set = voltage_to_set
        self._hdf5_storage_path = hdf5_storage_path

    def tuning_complete(self) -> bool:
        if self._current_tuner_index == len(self._tuning_hierarchy):
            return True
        else:
            return False

    def ready_to_tune(self) -> bool:
        naming_coherent = True
        all_gates = set(self._experiment.read_gate_voltages().index)
        if self._voltage_to_set is not None:
            assert set(self._voltage_to_set.index).issubset(all_gates)
        for tuner_number, par_tuner in enumerate(self._tuning_hierarchy):
            solver = par_tuner.solver
            if par_tuner.last_voltages is not None:
                if not set(par_tuner.last_voltages.index).issubset(all_gates):
                    print("The following gates are not known to the experiment but used in last_voltage in the "
                          "creation of the parameter tuner number" + str(tuner_number))
                    for gate in set(par_tuner.last_voltages.index) - all_gates:
                        print(gate)
                    naming_coherent = False

            if not set(par_tuner.solver.current_position.index).issubset(all_gates):
                print("The following gates are not known to the experiment but used in current_position in the "
                      "creation of the solver of parameter tuner number" + str(tuner_number))
                for gate in set(par_tuner.solver.current_position.index) - all_gates:
                    print(gate)
                naming_coherent = False

            if isinstance(solver, NewtonSolver):
                for gradient_estimator in solver.gradient_estimators:
                    if gradient_estimator.current_position is not None:
                        if not set(gradient_estimator.current_position.index).issubset(all_gates):
                            print("The following gates are not known to the experiment but used in current_position in"
                                  "the gradient estimator ")
                            print(gradient_estimator)
                            for gate in set(gradient_estimator.current_position) - all_gates:
                                print(gate)
                                naming_coherent = False

                    if isinstance(par_tuner, SubsetTuner):
                        if not set(par_tuner.tunable_gates).issubset(set(gradient_estimator.current_position.index)):
                            print("The following gates are to be tuned by the SubsetTuner number " + str(tuner_number)
                                  + " but they dont appear in the current positions of its gradient estimator")
                            print(gradient_estimator)
                            for gate in set(par_tuner.tunable_gates) - set(gradient_estimator.current_position.index):
                                print(gate)
                                naming_coherent = False

        return naming_coherent

    def get_current_tuner(self):
        return self._tuning_hierarchy[self._current_tuner_index]

    def iterate(self):
        if self._voltage_to_set is not None:
            self._experiment.set_gate_voltages(self._voltage_to_set)
            self._current_tuner_index = 0
            self._voltage_to_set = None
        elif not self._current_tuner_status:
            if self.get_current_tuner().is_tuned(self._experiment.read_gate_voltages()):
                self._current_tuner_index += 1
            else:
                self._current_tuner_status = True
        else:
            self._voltage_to_set = self.get_current_tuner().get_next_voltages()
            self._current_tuner_status = False

    def autotune(self):
        if not self.ready_to_tune():
            print("Setup incomplete!")
            return

        tuning_storage_path = self._hdf5_storage_path + r"\\" + time_string()
        os.makedirs(name=tuning_storage_path)

        while not self.tuning_complete():
            self.iterate()

            filename = tuning_storage_path + r"\\" + time_string() + ".hdf5"
            hdf5_file = h5py.File(filename, 'w-')
            to_hdf5(hdf5_file, name="autotuner", obj=self, reserved={"experiment": self._experiment})

    def to_hdf5(self):
        return dict(
            experiment=self._experiment,
            tuning_hierarchy=self._tuning_hierarchy,
            current_tuner_index=self._current_tuner_index,
            current_tuner_status=self._current_tuner_status,
            voltage_to_set=self._voltage_to_set,
            hdf5_storage_path=self._hdf5_storage_path
        )


def load_auto_tuner(file, reserved) -> Autotuner:
    assert "experiment" in reserved
    hdf5_handle = h5py.File(file, mode="r")
    loaded_data = from_hdf5(hdf5_handle, reserved=reserved)
    return loaded_data["autotuner"]
