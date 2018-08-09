# -*- coding: utf-8 -*-
import copy
from typing import Any, Union

import numpy as np

from .achrive import Achrive
from .simulated_annealing_base import SimulatedAnnealingBase
from ff_optimum.cores.utilities import EventLogger

__all__ = ['DominanceBasedMultiobjectiveSimulatedAnnealingOptimizer']

logger = EventLogger(__name__)


class DominanceBasedMultiobjectiveSimulatedAnnealingOptimizer(
        SimulatedAnnealingBase):

    """
    Class for the implementation of the dominance based multi-objective
    simulated annealing algorithm

    Attributes
    ----------
    __achrive: Achrive

    Methods
    -------
    optimize
        perform optimization

    save_parameters_to_file
        save the solution in the achrive to file

    """

    __slots__ = ['__achrive']

    def __init__(self, number_of_processors: int, profile: str,
                 package_name: str, package_settings: dict,
                 parameters: Union[dict, np.ndarray],
                 constraints_source: str, constraints_input: Any,
                 command_holder_train: object, training_data: dict,
                 plot_information: dict, alogrithm_parameters: dict,
                 output_directory: str):

        logger.info('Algorithm name: Dominance Based Multiobjective '
                    'Simulated Annealing')

        super(DominanceBasedMultiobjectiveSimulatedAnnealingOptimizer,
              self).__init__(
            number_of_processors, profile, package_name, package_settings,
            parameters, constraints_source, constraints_input,
            command_holder_train, training_data, plot_information,
            alogrithm_parameters, output_directory)

        __achrive_size = alogrithm_parameters.get('achrive_size', 50)

        logger.info(f'Achrive size: {__achrive_size}')

        self.__achrive = Achrive(__achrive_size)

        self.__fill_achrive(alogrithm_parameters.get('reduction', ["", 0]))

    def optimize(self) -> None:
        """
        Run the dominance based simulated annealing algorithm

        Returns
        -------
        None

        See Also
        -------
        _one_epoch
        """

        for epoch in range(self._number_of_epoch):
            self._one_epoch()

    def _one_epoch(self):
        """
        run one epoch of optimization

        Parameters
        ----------
        None

        Returns
        -------
        __achrive: Achrive
            object containing non-dominated solutions
        """

        next_parameter_generator = self._package.next_parameter_generator

        logger.info(f'Optimization starts')

        current_parameters_value = copy.deepcopy(self._parameters)

        __calcaulated_values = (
            self._commands_holder_train.execute_commands(
                current_parameters_value))

        current_fitness = self._evaluate_fitness(
            __calcaulated_values, self._training_dataset)

        consecutive_stop = 0

        for current_temperature, step in self._temperature_generator():

            logger.info(f'Temperature: {current_temperature:.4f}')

            number_of_acceptance, number_of_trial = 0, 0

            for idx, parameter, step_size, constraint in (
                next_parameter_generator(
                    current_parameters_value, self._step_size,
                    self._constraints, True)):

                number_of_trial += 1

                accepted, current_parameters_value, current_fitness = (
                    self.__one_move(current_parameters_value,
                                    current_fitness, idx,
                                    current_temperature, parameter,
                                    step_size, constraint))

                logger.info(f'Temperature: {current_temperature} '
                            f'Step: {step} '
                            f'Number of trial: {number_of_trial} '
                            f'accepted: {accepted}')

                logger.info(f'{current_fitness.objectives_values}')

                if accepted:
                    number_of_acceptance += 1

            if number_of_acceptance == 0:

                consecutive_stop += 1

                logger.info(f'number of consecutive stops remains: '
                            f'{self._number_of_stops - consecutive_stop}')

                if consecutive_stop > self._number_of_stops:
                    break
            else:
                consecutive_stop = 0

            if self._beta is None or self._beta < 1e-9:
                self._find_beta(number_of_trial, self._changes_in_error)

        logger.info('Optimization finishes')

        return self.__achrive

    def __one_move(self, current_parameters_value: np.ndarray,
                   current_fitness: object, idx: int,
                   current_temperature: float,
                   parameter: np.ndarray, step_size: float,
                   constraint: tuple) -> Union[bool, np.ndarray, float]:
        """
        Move one parameter and then evaluate fitness.
        If the number of being dominated in the achrive is lower than
        the current solution, or the new solution is accepted by the
        metrophoils criterion, update the current solution as new solution
        Add the new parameter set to the achrive if it is not dominated
        by any solution in the achrive

        Parameters
        ----------
        current_parameters_value
            the whole current parameter set
        current_fitness
            fitness value of the current solution
        idx
            index of the parameter inside the array
        current_temperature
            current temperature for metrophils criterion
        parameter
            the array of parameter to be moved
        step_size
            step size of the parameter
        constraint
            maximum and minimum value of the parameter

        Returns
        -------
        accepted
            whether the solution is accepted by metorphoils criterion
        current_parameters_value
            the updated parameter set
        current_fitness
            the updated fitness
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

        current_energy = self.__achrive.count_being_dominated_in_achrive(
            current_fitness)

        new_energy = self.__achrive.count_being_dominated_in_achrive(
            new_fitness)

        change_in_energy = ((new_energy - current_energy) /
                            self.__achrive.get_achrive_size())

        accepted = self._metropolis_criteria(change_in_energy,
                                             current_temperature)

        if self._beta is None or self._beta < 1e-9:
            self._changes_in_error.append(change_in_energy)

        if accepted:

            current_fitness = new_fitness

            if new_energy == 0:

                logger.info('Solution is added to the achrive')

                solution = {
                    'parameter': copy.deepcopy(current_parameters_value),
                    'fitness': new_fitness}

                self.__achrive.pop_dominated_solution(solution)

                self.__achrive.push_solution(solution)

                self.__achrive.generate_attainment_surface()

        else:
            parameter[idx] = rollback_value

        return accepted, current_parameters_value, current_fitness

    def save_error_to_file(self):
        pass

    def save_parameters_to_file(self, filename: str) -> None:
        """
        Save the parameters in the achrive to file

        Parameters
        ----------
        filename
            no use just to compact wtih the base class

        Returns
        -------
        None

        See Also
        --------
        Achrive.dump_parameters_to_file
        """

        self.__achrive.dump_parameters_to_file(
            self._package, self._output_directory)

    def __fill_achrive(self, reduction: list, target_step: int=5) -> None:
        """
        Fill in the achrive by running number of target step

        Parameters
        ----------
        reduction
            information of the reduction scheme

        target_step
            number of step to be run for filling the achrive

        Returns
        -------
        None

        See Also
        --------
        __dimension_reduction
        Achrive.push_solution
        """

        logger.info('Start filling achrive')

        next_parameter_generator = self._package.next_parameter_generator

        parameters = copy.deepcopy(self._parameters)

        for step in range(target_step):

            for idx, parameter, step_size, constraint in (
                next_parameter_generator(
                    parameters, self._step_size,
                    self._constraints, True)):

                parameter[idx] *= 1 + (np.random.uniform(-1, 1) * step_size)

                parameter[idx] = min(max(parameter[idx], constraint[0]),
                                     constraint[1])

                __calcaulated_values = (
                    self._commands_holder_train.execute_commands(parameters))

                fitness = self._evaluate_fitness(
                    __calcaulated_values, self._training_dataset)

                solution = {'parameter': copy.deepcopy(parameters),
                            'fitness': fitness}

                self.__achrive.push_solution(solution)

            logger.info('One step is computed')

        if reduction[1]:
            self.__dimension_reduction(reduction)

        logger.info(f'Achrive is filled, size: '
                    f'{self.__achrive.get_achrive_size()}')

    def __dimension_reduction(self, reduction: list) -> None:
        """
        Dimension reduction of objectives

        Parameters
        ----------
        reduction
            information of the reduction scheme

        Returns
        -------
        None

        See Also
        --------
        Achrive.dimension_reduction
        """

        for number_of_reduction in range(reduction[1]):

            objective_names = self.__achrive.get_objectives_names()

            objective_by_mol, objective_to_be_poped_by_mol = {}, {}

            for objective_name in objective_names:
                mol_name = objective_name.split('_')[0]

                if objective_by_mol.get(mol_name, None) is None:
                    objective_by_mol[mol_name] = []

                objective_by_mol[mol_name].append(objective_name)

            objectives_names_to_be_poped = (
                self.__achrive.dimension_reduction(reduction[0]))

            for name in objectives_names_to_be_poped:

                mol = name.split('_')[0]

                if objective_to_be_poped_by_mol.get(mol, None) is None:
                    objective_to_be_poped_by_mol[mol_name] = []

                objective_to_be_poped_by_mol[mol_name].append(name)

                remove_mol = self._package.pop_objective_from_settings(
                    self._package_setting_train, name)

                if not remove_mol:
                    objective_to_be_poped_by_mol[mol_name].remove(name)

            for key in objective_to_be_poped_by_mol.keys():
                if (set(objective_by_mol[key]) ==
                        set(objective_to_be_poped_by_mol[key])):
                    self._commands_holder_train.compiled_commands.pop(key)
                    logger.info(f'{key} is poped from commands')
