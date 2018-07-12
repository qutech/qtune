import h5py
import os
import os.path
import pandas as pd
from qtune.util import time_string
from qtune.experiment import Experiment
from typing import List, Optional
from qtune.parameter_tuner import ParameterTuner, SubsetTuner
from qtune.solver import NewtonSolver
from qtune.storage import to_hdf5, HDF5Serializable, from_hdf5, AsynchronousHDF5Writer
import logging


class Autotuner(metaclass=HDF5Serializable):
    """
    The auto tuner class combines the evaluator and solver classes to tune an experiment.
    """

    def __init__(self, experiment: Experiment, tuning_hierarchy: List[ParameterTuner] = None,
                 current_tuner_index: int = 0, current_tuner_status: bool = False,
                 voltage_to_set: Optional[pd.Series] = None, hdf5_storage_path: Optional[str] = None,
                 append_time_to_path: bool=True):
        self._experiment = experiment
        self._tuning_hierarchy = tuning_hierarchy
        for par_tuner in tuning_hierarchy:
            if par_tuner.last_voltages is None:
                par_tuner._last_voltage = self._experiment.read_gate_voltages()
        self._current_tuner_index = current_tuner_index
        self._current_tuner_status = current_tuner_status
        self._voltage_to_set = voltage_to_set

        if hdf5_storage_path:
            if append_time_to_path:
                self._hdf5_storage_path = os.path.join(hdf5_storage_path, time_string())
            else:
                self._hdf5_storage_path = hdf5_storage_path
        else:
            self._hdf5_storage_path = None
        self._asynchrone_writer = None
        self._logger = 'qtune'

    @property
    def asynchrone_writer(self):
        if self._asynchrone_writer is None:
            self._asynchrone_writer = AsynchronousHDF5Writer(reserved={"experiment": self._experiment},
                                                             multiprocess=False)
        return self._asynchrone_writer

    @property
    def logger(self):
        return logging.getLogger(self._logger)

    @logger.setter
    def logger(self, val: str):
        assert isinstance(val, str)
        self._logger = val

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

    def is_tuning_complete(self) -> bool:
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

    def __getstate__(self):
        """Do not pickle the async writer object"""
        state = self.__dict__.copy()
        state['_asynchrone_writer'] = None
        return state

    def save_current_status(self):
        if self._hdf5_storage_path:
            if not os.path.isdir(self._hdf5_storage_path):
                os.makedirs(self._hdf5_storage_path)
            filename = os.path.join(self._hdf5_storage_path, time_string() + ".hdf5")
            self.asynchrone_writer.write(self, file_name=filename, name='autotuner')
            # hdf5_file = h5py.File(filename, 'w-')
            # to_hdf5(hdf5_file, name="autotuner", obj=self, reserved={"experiment": self._experiment})

    def get_current_tuner(self):
        return self._tuning_hierarchy[self._current_tuner_index]

    def iterate(self):
        if not self.ready_to_tune():
            raise RuntimeError('The setup of the Autotuner class is incomplete!')

        if self.is_tuning_complete():
            raise RuntimeError('The tuning is already complete!')

        if self._voltage_to_set is not None:
            if self.voltages_to_set.isna().any():
                raise RuntimeError('A voltage is required to be set to NAN')

            voltage_state_change = pd.DataFrame()
            voltage_state_change['current'] = self._tuning_hierarchy[0].last_voltages[self._voltage_to_set.index]
            voltage_state_change['target']  = self._voltage_to_set
            voltage_state_change['step']    = voltage_state_change['target'] - voltage_state_change['current']
            self.logger.info("The voltages will be changed by:\n{}".format(
                voltage_state_change
            ))

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

        self.save_current_status()

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
