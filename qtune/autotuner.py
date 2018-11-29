import h5py
import os
import os.path
import pandas as pd
from qtune.util import time_string
from qtune.experiment import Experiment
from typing import List, Optional, Dict
from qtune.parameter_tuner import ParameterTuner, SubsetTuner
from qtune.solver import NewtonSolver
from qtune.storage import HDF5Serializable, from_hdf5, AsynchronousHDF5Writer
import logging


class Autotuner(metaclass=HDF5Serializable):
    """
    The Autotuner class manages the communication between the ParameterTuner classes and communicates with the
    experiment by setting and getting voltages for the ParameterTuner classes. The ParameterTuner classes are structured
    in a hierarchy to take their interdependency into account.
    """

    def __init__(self, experiment: Experiment, tuning_hierarchy: List[ParameterTuner] = None,
                 current_tuner_index: int = 0, current_tuner_status: bool = False,
                 voltage_to_set: Optional[pd.Series] = None, hdf5_storage_path: Optional[str] = None,
                 append_time_to_path: bool = True):
        """
        Initialize the AutoTuner.

        :param experiment: The experiment which shall be tuned(e.g. gate-defined quantum dots. ).

        :param tuning_hierarchy: The tuning_hierarchy takes the interdependency of parameter groups into account. Each
        ParameterTuner represents a group of parameters. The Hierarchy is saved in a list where the first element in the
        list is on the lowest position of the hierarchy. The ParameterTuner in the hierarchy are tuned from bottom to
        the top. Each ParameterTuner can be given information about the ParameterTuner which come below in the
        hierarchy. The Autotuner works in solving iterations where each iteration ends when new voltages are calculated.
        The Autotuner starts at the bottom of the hierarchy in each iteration, going upwards when a parameter is
        evaluated until a ParameterTuner suggest new voltages. The Autotuner is always at a specific position in the
        hierarchy.

        :param current_tuner_index: Hierarchy number of the parameter currently tuned.

        :param current_tuner_status: True if the ParameterTuner at the current position in the hierarchy has evaluated
        its parameters in the current iteration.

        :param voltage_to_set: Voltages to be set in the next iteration.

        :param hdf5_storage_path: Path to the HDF5 library where the autotuner is saved.

        :param append_time_to_path: True if the current time is to be appended to the saving path.
        """
        self._experiment = experiment
        self._tuning_hierarchy = tuning_hierarchy
        for par_tuner in tuning_hierarchy:
            if par_tuner.last_voltages is None:
                par_tuner._last_voltage = self._experiment.read_gate_voltages()
        self._current_tuner_index = current_tuner_index
        self._current_tuner_status = current_tuner_status
        self._voltages_to_set = voltage_to_set

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
        """
        The asynchrone writer is only initialized on demand.
        :return:
        """
        if self._asynchrone_writer is None:
            self._asynchrone_writer = AsynchronousHDF5Writer(reserved={"experiment": self._experiment},
                                                             multiprocess=False)
        return self._asynchrone_writer

    @property
    def logger(self):
        """
        The Autotuner saves only the name of the logger. The AutoTuner gets the actual object on demand.
        :return:
        """
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
        return self._voltages_to_set

    @property
    def current_tuner_status(self):
        return self._current_tuner_status

    @property
    def all_estimators_ready(self):
        """
        Checks if any gradient estimator requires a measurement for the gradient estimation.
        :return: True if no GradientEstimator needs a measurement to determine its gradient.
        """
        for par_tuner in self.tuning_hierarchy:
            solver = par_tuner.solver
            if isinstance(solver, NewtonSolver):
                for grad_est in solver.gradient_estimators:
                    if isinstance(grad_est.require_measurement(solver.current_position.index), pd.Series):
                        if not grad_est.require_measurement(solver.current_position.index).empty:
                            return False
        return True

    @property
    def tuned_parameters(self):
        tuned_parameters = set()
        for i in range(0, self.current_tuner_index):
            not_na_index = self.tuning_hierarchy[i].target.drop(
                [ind for ind in self.tuning_hierarchy[i].target.columns if
                 self.tuning_hierarchy[i].target.isna().all()[ind]], axis='columns').dropna().index
            tuned_parameters = tuned_parameters.union(set(not_na_index))
        return tuned_parameters

    def is_tuning_complete(self) -> bool:
        if self._current_tuner_index == len(self._tuning_hierarchy):
            return True
        else:
            return False

    def ready_to_tune(self) -> bool:
        """
        Verifies that gates/voltages and parameters are named coherently.
        :return: True if the naming is coherent.
        """
        naming_coherent = True
        all_gates = set(self._experiment.read_gate_voltages().index)
        if self._voltages_to_set is not None:
            assert set(self._voltages_to_set.index).issubset(all_gates)
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

    def change_targets(self, target_changes: List[Dict[str, pd.Series]]):
        """
        This function changes the targets to continue the tuning towards a new goal.
        :param target_changes: A list corresponding to the tuning hierarchy. The list elements contain dicts
        corresponding to the categories of the target like 'desired' or 'tolerance'.
        :return: None
        """
        assert (len(target_changes) <= len(self.tuning_hierarchy))
        for i, target_change in enumerate(target_changes):
            self.tuning_hierarchy[i].target = target_change
        self._current_tuner_index = 0
        self._voltages_to_set = None

    def __getstate__(self):
        """Do not pickle the async writer object"""
        state = self.__dict__.copy()
        state['_asynchrone_writer'] = None
        return state

    def save_current_status(self):
        """
        Writes the current state to the HDF5 library.
        :return: None
        """
        if self._hdf5_storage_path:
            if not os.path.isdir(self._hdf5_storage_path):
                os.makedirs(self._hdf5_storage_path)
            filename = os.path.join(self._hdf5_storage_path, time_string() + ".hdf5")
            self.asynchrone_writer.write(self, file_name=filename, name='autotuner')
            # hdf5_file = h5py.File(storage_path, 'w-')
            # to_hdf5(hdf5_file, name="autotuner", obj=self, reserved={"experiment": self._experiment})

    def get_current_tuner(self):
        return self._tuning_hierarchy[self._current_tuner_index]

    def iterate(self):
        """
        Execute one iteration. This can either be a measurement and the subsequent extraction of a parameter, the
        calculation of the next voltages or setting the next voltages.
        Afterwards the state is written to the HDF5 library.
        :return: None
        """
        if not self.ready_to_tune():
            raise RuntimeError('The setup of the Autotuner class is incomplete!')

        if self.is_tuning_complete():
            raise RuntimeError('The tuning is already complete!')

        if self._voltages_to_set is not None:
            if self.voltages_to_set.isna().any():
                raise RuntimeError('A voltage is required to be set to NAN')

            voltage_state_change = pd.DataFrame()
            voltage_state_change['current'] = self._tuning_hierarchy[0].last_voltages[self._voltages_to_set.index]
            voltage_state_change['target'] = self._voltages_to_set
            voltage_state_change['step'] = voltage_state_change['target'] - voltage_state_change['current']
            self.logger.info("The voltages will be changed by:\n{}".format(
                voltage_state_change
            ))

            self._experiment.set_gate_voltages(self._voltages_to_set)
            self._current_tuner_index = 0
            self._voltages_to_set = None
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
            self._voltages_to_set = self.get_current_tuner().get_next_voltages(tuned_parameters=self.tuned_parameters)
            self._current_tuner_status = False
            self.logger.info("Next voltages are being calculated.")

        self.save_current_status()

    def restart(self):
        """ Reads new voltages and communicates them to the member classes. This function can be called when the
        experiment has been configured manually."""
        self._current_tuner_index = 0
        self._voltages_to_set = None
        current_voltages = self._experiment.read_gate_voltages()
        for par_tuner in self.tuning_hierarchy:
            par_tuner.restart(current_voltages)

    def to_hdf5(self):
        return dict(
            experiment=self._experiment,
            tuning_hierarchy=self._tuning_hierarchy,
            current_tuner_index=self._current_tuner_index,
            current_tuner_status=self._current_tuner_status,
            voltage_to_set=self._voltages_to_set,
            hdf5_storage_path=self._hdf5_storage_path
        )

    def __repr__(self):
        return "{type}({data})".format(type=type(self), data=self.to_hdf5())


def load_auto_tuner(file, reserved) -> Autotuner:
    """
    Loads an Autotuner class out of the HDF5 library.
    :param file: File containing the HDF5 library.
    :param reserved: Reserved objects.
    :return: The reloaded Autotuner.
    """
    assert "experiment" in reserved
    hdf5_handle = h5py.File(file, mode="r")
    loaded_data = from_hdf5(hdf5_handle, reserved=reserved)
    return loaded_data["autotuner"]
