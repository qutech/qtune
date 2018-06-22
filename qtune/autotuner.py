import h5py
import os
import os.path
import pandas as pd
from qtune.util import time_string
from qtune.experiment import Experiment
from typing import List, Optional, Sequence
from qtune.parameter_tuner import ParameterTuner, SubsetTuner
from qtune.solver import NewtonSolver
from qtune.storage import to_hdf5, HDF5Serializable, from_hdf5
import matplotlib.pyplot as plt
import matplotlib
import logging


class Autotuner(metaclass=HDF5Serializable):
    """
    The auto tuner class combines the evaluator and solver classes to tune an experiment.
    """

    def __init__(self, experiment: Experiment, tuning_hierarchy: List[ParameterTuner] = None,
                 current_tuner_index: int = 0, current_tuner_status: bool = False,
                 voltage_to_set: Optional[pd.Series] = None, hdf5_storage_path: str = None):
        self._experiment = experiment
        self._tuning_hierarchy = tuning_hierarchy
        for par_tuner in tuning_hierarchy:
            if par_tuner.last_voltages is None:
                par_tuner._last_voltage = self._experiment.read_gate_voltages()
        self._current_tuner_index = current_tuner_index
        self._current_tuner_status = current_tuner_status
        self._voltage_to_set = voltage_to_set
        self._hdf5_storage_path = os.path.join(hdf5_storage_path, time_string())
        os.makedirs(name=self._hdf5_storage_path)

        self.logger = logging.getLogger(name="autotuner_logger")
        self.console_handler = logging.Handler(level=logging.DEBUG)
        self.logger.addHandler(self.console_handler)

    @property
    def tuning_hierarchy(self):
        return self._tuning_hierarchy

    @property
    def current_tuner_index(self):
        return self._current_tuner_index

    @property
    def voltages_to_set(self):
        return self._voltage_to_set

    @property
    def current_tuner_status(self):
        return self._current_tuner_status

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
                    self.logger.error("The following gates are not known to the experiment but used in last_voltage in"
                                      "the creation of the parameter tuner number " + str(tuner_number))
                    self.logger.error(set(par_tuner.last_voltages.index) - all_gates)
                    naming_coherent = False

            if not set(par_tuner.solver.current_position.index).issubset(all_gates):
                self.logger.error("The following gates are not known to the experiment but used in current_position in"
                                  "the creation of the solver of parameter tuner number" + str(tuner_number))
                self.logger.error(set(par_tuner.solver.current_position.index) - all_gates)
                naming_coherent = False

            if isinstance(solver, NewtonSolver):
                for gradient_estimator in solver.gradient_estimators:
                    if gradient_estimator.current_position is not None:
                        if not set(gradient_estimator.current_position.index).issubset(all_gates):
                            self.logger.error("The following gates")
                            self.logger.error(set(gradient_estimator.current_position) - all_gates)
                            self.logger.error("are not known to the experiment but used in current position in the"
                                              "gradient estimator ")
                            self.logger.error(gradient_estimator)
                            naming_coherent = False

                    if isinstance(par_tuner, SubsetTuner):
                        if not set(par_tuner.tunable_gates).issubset(set(gradient_estimator.current_position.index)):
                            self.logger.error("The following gates")
                            self.logger.error(set(par_tuner.tunable_gates) -
                                              set(gradient_estimator.current_position.index))
                            self.logger.error("are to be tuned by the SubsetTuner number " + str(tuner_number) +
                                              " but they do not appear in the current positions of its gradient"
                                              "estimator")
                            self.logger.error(gradient_estimator)
                            naming_coherent = False
        return naming_coherent

    def get_current_tuner(self):
        return self._tuning_hierarchy[self._current_tuner_index]

    def iterate(self):
        if not self.ready_to_tune():
            raise RuntimeError('The setup of the Autotuner class is incomplete!')

        if self.tuning_complete():
            raise RuntimeError('The tuning is already complete!')

        if self._voltage_to_set is not None:
            self.logger.info("The voltages will be changed by:")
            self.logger.info(self._voltage_to_set - self._tuning_hierarchy[0].last_voltages[self._voltage_to_set.index])
            self._experiment.set_gate_voltages(self._voltage_to_set)
            self._current_tuner_index = 0
            self._voltage_to_set = None
        elif not self._current_tuner_status:
            self.logger.info("The parameters of ParameterTuner number " + str(self._current_tuner_index) +
                             " are being evaluated.")
            if self.get_current_tuner().is_tuned(self._experiment.read_gate_voltages()):
                self._current_tuner_index += 1
                self.logger.info("The parameters are tuned. Move on to ParameterTuner number " +
                                 str(self._current_tuner_index))
            else:
                self._current_tuner_status = True
                self.logger.info("The parameters are not tuned yet.")
                if self.get_current_tuner().target["desired"].notna().all():
                    self.logger.info("The distance to their target is: ")
                    self.logger.info(self.get_current_tuner().target["desired"] -
                                     self.get_current_tuner().last_parameters_and_variances[0]
                                     [self.get_current_tuner().target.index])
        else:
            self._voltage_to_set = self.get_current_tuner().get_next_voltages()
            self._current_tuner_status = False
            self.logger.info("Next voltages are being calculated.")

        filename = os.path.join(self._hdf5_storage_path, time_string() + ".hdf5")
        hdf5_file = h5py.File(filename, 'w-')
        to_hdf5(hdf5_file, name="autotuner", obj=self, reserved={"experiment": self._experiment})

    def autotune(self):

            """
            if self._live_plotting:
                if self._reader is None:
                    self._reader = Reader(self._tuning_hierarchy)
                else:
                    self._reader.append_tuning_hierarchy(tuning_hierarchy=self._tuning_hierarchy)

                if self._plotting_objects is not None:
                    for obj in self._plotting_objects:
                        if isinstance(obj, matplotlib.figure.Figure):
                            plt.close(obj)
                self._plotting_objects = self._reader.plot_tuning(voltage_indices=self._plotted_gates,
                                                                  parameter_names=self._plotted_parameters,
                                                                  gradient_names=self._plotted_gradients,
                                                                  mode=self._plotting_mode)
                plt.pause(5)
            """

    def to_hdf5(self):
        return dict(
            experiment=self._experiment,
            tuning_hierarchy=self._tuning_hierarchy,
            current_tuner_index=self._current_tuner_index,
            current_tuner_status=self._current_tuner_status,
            voltage_to_set=self._voltage_to_set,
            hdf5_storage_path=self._hdf5_storage_path
        )

    def __repr__(self):
        return "{type}({data})".format(type=type(self), data=self.to_hdf5())


def load_auto_tuner(file, reserved) -> Autotuner:
    assert "experiment" in reserved
    hdf5_handle = h5py.File(file, mode="r")
    loaded_data = from_hdf5(hdf5_handle, reserved=reserved)
    return loaded_data["autotuner"]
