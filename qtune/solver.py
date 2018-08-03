from typing import Tuple, Sequence, Deque, Callable, Optional, Dict
import enum
import math
import logging

import numpy as np
import pandas as pd

from qtune.gradient import GradientEstimator
from qtune.storage import HDF5Serializable
from qtune.util import get_orthogonal_vector


def make_target(desired: pd.Series = np.nan,
                maximum: pd.Series = np.nan,
                minimum: pd.Series = np.nan,
                tolerance: pd.Series = np.nan,
                cost_threshold: pd.Series = np.nan,
                rescaling_factor: pd.Series = np.nan):
    for ser in (desired, maximum, minimum, tolerance):
        if isinstance(ser, pd.Series):
            names = ser.index
            break
    else:
        raise RuntimeError('Could not extract values names from arguments')

    def to_series(arg):
        if not isinstance(arg, pd.Series):
            return pd.Series(arg, index=names)
        else:
            return arg[names]

    if not isinstance(rescaling_factor, pd.Series):
        if rescaling_factor != rescaling_factor:
            rescaling_factor = pd.Series(data=1, index=names)

    return pd.DataFrame({'desired': to_series(desired),
                         'minimum': to_series(minimum),
                         'maximum': to_series(maximum),
                         'tolerance': to_series(tolerance),
                         'cost_threshold': to_series(cost_threshold),
                         'rescaling_factor': to_series(rescaling_factor)},
                        index=names)


class Solver(metaclass=HDF5Serializable):
    """
    The solver class implements an algorithm to minimise the difference of the values to the target values.
    """
    _current_position = None
    _current_values = None
    _target = None
    _logger = 'qtune'

    @property
    def current_position(self) -> pd.Series:
        return self._current_position

    @property
    def logger(self):
        return logging.getLogger(self._logger)

    def suggest_next_position(self, tuned_parameters=None) -> pd.Series:
        raise NotImplementedError()

    def update_after_step(self, position: pd.Series, values: pd.Series, variances: pd.Series):
        raise NotImplementedError()

    def to_hdf5(self):
        raise NotImplementedError()

    @property
    def target(self) -> pd.DataFrame:
        return self._target

    @target.setter
    def target(self, changes: Dict[str, pd.Series]):
        for category in changes:
            if not changes[category].index.isin(self.target[category].index).all():
                self.logger.error('The new target %s %s is not consistent with the previous one!'
                                  % (category, changes[category].index.difference(self.target[category].index)))
            if len(self.target.loc[changes[category].index, category]) != changes[category].size:
                self.logger.error('The new target %s %s is not consistent with the previous one! It does not have the '
                                  'right number of entries.'
                                  % (category, changes[category].index.difference(self.target[category].index)))
            self.target.loc[changes[category].index, category] = changes[category]

    @property
    def state(self) -> pd.Series:
        raise NotImplementedError()

    def rescale_values(self, values: pd.Series, variances: pd.Series):
        """The values are rescaling right after they have been given to the Solver. The target must therefore hold
        rescaled value."""
        rescaling_factor = self.target['rescaling_factor'].append(
            pd.Series(index=values.index.difference(self.target['rescaling_factor'].index), data=1))
        values /= rescaling_factor[values.index]
        variances /= rescaling_factor[values.index] ** 2

    def descale_values(self, values: pd.Series, variances: pd.Series):
        """The values are rescaling right after they have been given to the Solver. The target must therefore hold
        rescaled value."""
        rescaling_factor = self.target['rescaling_factor'].append(
            pd.Series(index=values.index.difference(self.target['rescaling_factor'].index), data=1))
        values *= rescaling_factor[values.index]
        variances *= rescaling_factor[values.index] ** 2


class NewtonSolver(Solver):
    """This solver uses (an estimate of) the jacobian and solves by inverting it.(Newton's method)

    The jacobian is put together from the given gradient estimators

    TODO: The jacobian is not automatically in the correct basis
    """

    def __init__(self, target: pd.DataFrame,
                 gradient_estimators: Sequence[GradientEstimator],
                 current_position: pd.Series,
                 current_values: pd.Series = None):
        self._target = target
        self._gradient_estimators = list(gradient_estimators)
        assert (len(self._target.index) == len(self._gradient_estimators))
        self._current_position = current_position
        for gradient_estimator in gradient_estimators:
            assert set(self._current_position.index).issubset(set(gradient_estimator.current_position.index))
        if current_values is not None:
            assert set(self.target.index).issubset(set(current_values.index))
            self._current_values = current_values[self._target.index]
        else:
            self._current_values = pd.Series(np.nan, index=self._target.index)
        assert len(self._current_position) >= len(self._current_values)

    @property
    def gradient_estimators(self):
        return self._gradient_estimators

    @property
    def jacobian(self) -> pd.DataFrame:
        gradients = [gradient.estimate() for gradient in self._gradient_estimators]
        return pd.concat(gradients, axis=1, keys=self._target.index).T

    @property
    def state(self):
        jacobian = self.jacobian[self._current_position.index]
        index = []
        for param in jacobian.index:
            for gate in jacobian.columns:
                index.append('d{}/d{}'.format(param, gate))
        return pd.Series(jacobian.values.ravel(), index=index)

    def suggest_next_position(self, tuned_parameters=None) -> pd.Series:
        tuned_parameters = set() if tuned_parameters is None else tuned_parameters
        tuned_estimators = [gradient
                            for i, gradient in enumerate(self._gradient_estimators)
                            if self.target.index[i] in tuned_parameters]
        tuned_gradients = [gradient.estimate() for gradient in tuned_estimators]
        if len(tuned_gradients) == 0:
            tuned_jacobian = None
        else:
            tuned_jacobian = pd.concat(tuned_gradients, axis=1).T

        for i, estimator in enumerate(self._gradient_estimators):
            if self.target.iloc[i].desired == self.target.iloc[i].desired:
                suggestion = estimator.require_measurement(gates=self._current_position.index,
                                                           tuned_jacobian=tuned_jacobian)
                if suggestion is not None and not suggestion.empty:
                    return suggestion

        if self._current_position is None:
            raise RuntimeError('NewtonSolver: Position not initialized.')

        # our jacobian is sufficiently accurate
        # nan targets are replaced with the current values

        required_diff = self.target.desired.add(-1 * self._current_values, fill_value=0).fillna(0)

        jacobian = self.jacobian[self._current_position.index]

        step, *_ = np.linalg.lstsq(jacobian, required_diff, rcond=None)

        # this is not exactly Newton

        return self._current_position + step

    def update_after_step(self, position: pd.Series, values: pd.Series, variances: pd.Series):
        for estimator, value, value_index, variance in zip(self._gradient_estimators,
                                                           values[self._current_values.index],
                                                           self._current_values.index, variances):
            if not math.isnan(self.target.desired[value_index]):
                estimator.update(position, value, variance, is_new_position=True)
        self._current_position = position[self._current_position.index]
        self._current_values = values[self._current_values.index]

    def to_hdf5(self):
        return dict(target=self.target,
                    gradient_estimators=self._gradient_estimators,
                    current_position=self._current_position,
                    current_values=self._current_values)

    def __repr__(self):
        return "{type}({data})".format(type=type(self), data=self.to_hdf5())


class NelderMeadSolver(Solver):
    class State(enum.Enum):
        start_iteration = 'start_iteration'
        build_up = 'built_up'
        reflect = 'reflect'
        expand = 'expand'
        contract_outside = 'contract_outside'
        contract_inside = 'contract_inside'
        shrink = 'shrink'

    def __init__(self,
                 target: pd.Series,
                 simplex: Sequence[Tuple[pd.Series, pd.Series]],
                 span: pd.Series,
                 current_position: pd.Series,
                 current_values: pd.Series,
                 reflected_point: Optional[Tuple[pd.Series, pd.Series]] = None,
                 shrunk_points: Optional[Sequence[Tuple[pd.Series, pd.Series]]] = None,
                 cost_function: Callable = lambda v: np.sum(v * v),
                 state='built_up'):
        """

        :param target:
        :param simplex:
        :param span: Used to build up the initial simplex
        :param cost_function:
        :param current_position:
        """
        if 'desired' not in target:
            raise ValueError('The NelderMead solver requires the desired column in the target.')
        self._target = target
        self._simplex = list(simplex)

        self._current_position = current_position
        self._current_values = current_values[self.target.desired.index]

        self._span = span[current_position.index]

        self._cost_function = cost_function

        self._state = self.State(state)

        self._reflected_point = reflected_point
        self._shrunk_points = shrunk_points or []

    @classmethod
    def from_scratch(cls,
                     target: pd.Series,
                     span: pd.Series,
                     current_position: pd.Series,
                     current_values: pd.Series):
        current_values = current_values[target.index]
        span = span[current_position.index]
        simplex = [(current_position, current_values)]

        return cls(target=target,
                   span=span,
                   current_position=current_position,
                   current_values=current_values,
                   simplex=simplex)

    @property
    def target(self):
        return self._target

    @property
    def simplex(self) -> Deque[Tuple[pd.Series, pd.Series]]:
        """Sorted chronologically if state=built_up otherwise sorted by cost"""
        return self._simplex

    def get_positions(self):
        return [position for position, _ in self.simplex]

    def cost_function(self, values: pd.Series):
        values = values - self.target.desired
        if self._cost_function:
            return self._cost_function(values)
        else:
            np.sum(values * values)

    def get_costs(self):
        return [self.cost_function(values) for _, values in self.simplex]

    @property
    def current_position(self) -> pd.Series:
        return self._current_position

    @property
    def current_values(self) -> pd.Series:
        return self._current_values

    def _get_sorted_simplex(self) -> list:
        def get_entry_cost(entry):
            _, values = entry
            return self.cost_function(values)

        return sorted(self.simplex, key=get_entry_cost)

    def _insert_into_simplex(self, position, values):
        self._simplex.append((position, values))
        self._simplex = self._get_sorted_simplex()
        self._simplex.pop()

    def suggest_next_position(self) -> pd.Series:
        positions = self.get_positions()
        worst_position = positions[-1]

        m = pd.DataFrame(positions).mean(axis=0)
        r = 2 * m - worst_position
        s = m + 2 * (m - worst_position)
        c = m + (r - m) / 2
        cc = m + (worst_position - m) / 2

        if len(self.simplex) < len(self.current_position) + 1:
            starting_point = self.simplex[0][0]
            diffs = [position - starting_point for position, values in self.simplex]
            self._state = self.State.build_up
            return starting_point + get_orthogonal_vector(diffs) * self._span

        elif self._state in (self.State.start_iteration, self.State.reflect):
            self._state = self.State.reflect
            return r

        elif self._state == self.State.expand:
            return s

        elif self._state == self.State.contract_outside:
            return c

        elif self._state == self.State.contract_inside:
            return cc

        elif self._state == self.State.shrink:
            i = len(self._shrunk_points) + 1
            if i == len(self.simplex):
                raise RuntimeError('Bug')

            v = positions[0] + (positions[i] - positions[0]) / 2
            return v

        else:
            raise RuntimeError('Bug')

    def update_after_step(self, position: pd.Series, values: pd.Series, variances: pd.Series):
        position = position[self._current_position.index]
        values = values[self._current_values.index]

        if len(self.simplex) < len(self.current_position) + 1:
            self.simplex.append((position, values))

            if len(self.simplex) == len(self.current_position) + 1:
                self._simplex = self._get_sorted_simplex()
                self._state = self.State.start_iteration

        elif self._state == self.State.reflect:
            new_cost = self.cost_function(values)
            costs = self.get_costs()

            self._reflected_point = (position, values)

            if new_cost < costs[0]:
                # require expansion point
                self._state = self.State.expand

            elif new_cost >= costs[-2]:
                if new_cost < costs[-1]:
                    self._state = self.State.contract_outside
                else:
                    self._state = self.State.contract_inside

            else:
                # costs[0] <= new_cost < costs[-2]
                self._insert_into_simplex(position, values)

        elif self._state == self.State.expand:
            reflected_pos, reflected_val = self._reflected_point
            reflection_cost = self.cost_function(reflected_val)

            expansion_cost = self.cost_function(values)
            if expansion_cost < reflection_cost:
                self._insert_into_simplex(position, values)
            else:
                self._insert_into_simplex(reflected_pos, reflected_val)

        elif self._state == self.State.contract_outside:
            reflected_pos, reflected_val = self._reflected_point
            reflection_cost = self.cost_function(reflected_val)

            contraction_cost = self.cost_function(values)
            if contraction_cost < reflection_cost:
                self._insert_into_simplex(position, values)

            else:
                self._state = self.State.shrink

        elif self._state == self.State.contract_inside:
            worst_cost = self.get_costs()[-1]

            contraction_cost = self.cost_function(values)
            if contraction_cost < worst_cost:
                self._insert_into_simplex(position, values)

            else:
                self._state = self.State.shrink

        elif self._state == self.State.shrink:
            self._shrunk_points.append((position, values))

            if len(self._shrunk_points) == len(self._current_position):
                for shrink_point in self._shrunk_points:
                    self._insert_into_simplex(*shrink_point)
                self._shrunk_points = []

        else:
            # we just got a point without requesting it
            worst_cost = self.get_costs()[-1]
            new_cost = self.cost_function(values)

            if new_cost < worst_cost:
                self._insert_into_simplex(position, values)

            self._state = self.State.start_iteration

        self._current_position = position
        self._current_values = values

    def to_hdf5(self):
        return dict(target=self._target,
                    simplex=self._simplex,
                    span=self._span,
                    current_position=self._current_position,
                    current_values=self._current_values,
                    reflected_point=self._reflected_point,
                    shrunk_points=self._shrunk_points,
                    cost_function=self._cost_function,
                    state=str(self._state))


class ForwardingSolver(Solver):
    """Solves by forwarding the values of the given values and renaming them to a voltage vector which updates the
    given position"""

    def __init__(self,
                 target: pd.DataFrame,
                 values_to_position: pd.Series,
                 current_position: pd.Series,
                 next_position: pd.Series = None):
        """

        :param values_to_position: A series of strings
        :param next_position:
        """
        self._target = target
        self._values_to_position = values_to_position
        self._current_position = current_position
        if next_position is None:
            next_position = self._current_position.copy()
        else:
            next_position = next_position[self._current_position.index]
        self._next_position = next_position

    def suggest_next_position(self, tuned_parameters) -> pd.Series:
        return self._next_position

    def update_after_step(self, position: pd.Series, values: pd.Series, variances: pd.Series):
        self._current_position[position.index] = position
        self._next_position[position.index] = position
        self.descale_values(values, variances)
        new_position_names = self._values_to_position[values.index].dropna()
        self._next_position[new_position_names] = values[new_position_names.index]

    def state(self):
        return pd.Series()

    def to_hdf5(self):
        return dict(target=self._target,
                    values_to_position=self._values_to_position,
                    current_position=self._current_position,
                    next_position=self._next_position)
