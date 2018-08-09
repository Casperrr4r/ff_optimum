# -*- coding: utf-8 -*-
import copy
import time
from typing import Any, Union

import numpy as np

from .simulated_annealing_base import SimulatedAnnealingBase
from ff_optimum.cores.utilities import EventLogger, file_set_path

__all__ = ['SimulatedAnnealingOptimizer']

logger = EventLogger(__name__)


class SimulatedAnnealingOptimizer(SimulatedAnnealingBase):

    """
    Class for the single object simulated annealing algorithm

    Attributes
    ----------
    __errors_trace: list
        List object contraining the error against temperature and steps
    """

    __slots__ = ['__errors_trace']

    def __init__(self, number_of_processors: int, profile: str,
                 package_name: str, package_settings: dict,
                 parameters: Union[dict, np.ndarray],
                 constraints_source: str, constraints_input: Any,
                 command_holder_train: object, training_data: dict,
                 plot_information: dict, algorithm_parameter: dict,
                 output_directory: str):

        logger.info('Algorithm name: Simulated Annealing')

        super(SimulatedAnnealingOptimizer, self).__init__(
            number_of_processors, profile, package_name, package_settings,
            parameters, constraints_source, constraints_input,
            command_holder_train, training_data, plot_information,
            algorithm_parameter, output_directory)

        self.__errors_trace = list()

    def optimize(self) -> Union[float, dict]:
        """
        Run the simulated annealing algorithm

        Returns
        -------
        best_error
            The global best error searched during optimzation

        best_param_values
            Parameters with the global best error searched during optimzation

        See Also
        -------
        _one_epoch
        """

        for epoch in range(self._number_of_epoch):
            best_err, best_param = self._one_epoch()

        return best_err, best_param

    def __one_move(self, current_parameters_value: np.ndarray,
                   current_fitness: float, idx: int,
                   current_temperature: float,
                   parameter: np.ndarray,
                   step_size: float,
                   constraint: tuple) -> Union[bool, np.ndarray, float]:
        """
        Move one of the parameter, evaluate the fitness and decide whether
        the new set of parameters to be accepted or not

        Parameters
        ----------
        current_parameters_value: dict
        current_fitness: float
        idx: int
        current_temperature: float
        parameter: dict
        step_size: dict
        constraint: dict

        Returns
        -------
        accepted
            True if the new set of parameters is accepted
            Otherwise False

        current_parameters_value
            New parameters value if the new set of parameters is accepted
            Otherwise pervious parameters value

        current_fitness
            New fitness if the new set of parameters is accepted
            Otherwise pervious fitness
        """

        rollback_value = parameter[idx]

        parameter[idx] *= 1 + (np.random.uniform(-1, 1) * step_size)

        parameter[idx] = min(max(parameter[idx], constraint[0]),
                             constraint[1])

        new_calculated_value = (
            self._commands_holder_train.execute_commands(
                current_parameters_value))

        new_fitness = self._evaluate_fitness(new_calculated_value,
                                             self._training_dataset)

        change_in_energy = new_fitness - current_fitness

        accepted = self._metropolis_criteria(change_in_energy,
                                             current_temperature)

        if self._beta is None:
            self._changes_in_error.append(change_in_energy)

        if accepted:
            current_fitness = new_fitness
        else:
            parameter[idx] = rollback_value

        return accepted, current_parameters_value, current_fitness

    def _one_epoch(self):
        """
        Run one epoch of simulated annealing

        Returns
        -------
        best_error
            The global best error searched during optimzation

        best_param_values
            Parameters with the global best error searched during optimzation

        See Also
        --------
        __one_move
        """

        error_dictionary = {'Temperature': [], 'Step': [], 'Trial': [],
                            'Error': []}

        try:

            next_parameter_generator = self._package.next_parameter_generator

            current_parameters_value = copy.deepcopy(self._parameters)

            current_calculated_value = (
                self._commands_holder_train.execute_commands(
                    current_parameters_value))

            current_error = self._evaluate_fitness(current_calculated_value,
                                                   self._training_dataset)

            logger.info(f'Optimization starts, '
                        f'initial error: {current_error:.4f}')

            best_param_values = copy.deepcopy(current_parameters_value)

            best_error = current_error

            for current_temperature, step in self._temperature_generator():

                logger.info(f'Temperature: {current_temperature:.4f}')

                number_of_acceptance, number_of_trial = 0, 0

                for idx, parameter, step_size, constraint in (
                    next_parameter_generator(
                        current_parameters_value, self._step_size,
                        self._constraints, True)):

                    if best_error < self._threshold:
                        raise StopIteration(
                            f'Best error {best_error:.4f} is smaller than '
                            f'threshold {self._threshold:.4f}')

                    number_of_trial += 1

                    accepted, current_parameters_value, current_error = (
                        self.__one_move(current_parameters_value,
                                        current_error, idx,
                                        current_temperature, parameter,
                                        step_size, constraint))

                    error_dictionary['Temperature'].append(current_temperature)

                    error_dictionary['Step'].append(step)

                    error_dictionary['Trial'].append(number_of_trial)

                    error_dictionary['Error'].append(current_error)

                    logger.info(f'Temperature: {current_temperature:.4f} '
                                f'Step: {step} Trial: {number_of_trial} '
                                f'Error: {current_error:.4f}')

                    if accepted:

                        number_of_acceptance += 1

                        if best_error > current_error:
                            if isinstance(best_param_values, np.ndarray):
                                np.copyto(best_param_values,
                                          current_parameters_value)
                            else:
                                best_param_values = copy.deepcopy(
                                    current_parameters_value)

                            best_error = current_error

                if self._beta is None:
                    self._find_beta(number_of_trial, self._changes_in_error)

        except StopIteration as e:

            logger.info(e)

            logger.info('Stop condition is achieved')

        finally:

            logger.info(f'Optimization finish best error: {best_error:.4f}')

            if isinstance(best_param_values, np.ndarray):
                np.copyto(self._parameters, best_param_values)
            else:
                self._parameters = copy.deepcopy(best_param_values)

            self.__errors_trace.append(error_dictionary)

            return best_error, copy.deepcopy(best_param_values)

    def save_error_to_file(self):
        """
        Save the error to the file

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        now = time.strftime('%y_%m_%d_%H_%M')

        with open('errors_sa_{now}', 'w+') as fp:

            epoch = 0

            fp.write(f'# epoch step trial temperature error')

            for trace in self.__errors_trace:

                for trial, step, temperature, error in zip(
                        trace['Trial'], trace['Step'],
                        trace['Temperature'], trace['Error']):
                    fp.write(f'{epoch} {step} {trial} {temperature} {error}')

                epoch += 1

    def save_parameters_to_file(self, filename='ffield_out') -> None:
        """
        Save the parameter to the file

        Parameters
        ----------
        filename
            filename of the parameters to be saved

        Returns
        -------
        None

        """

        file_path = file_set_path(self._output_directory, filename)

        self._package.save_parameters_to_file(self._parameters, file_path)
