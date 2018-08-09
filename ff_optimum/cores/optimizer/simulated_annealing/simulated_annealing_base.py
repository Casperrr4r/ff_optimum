# -*- coding: utf-8 -*-
import abc
from functools import reduce
import operator
from typing import Any, Union, Generator

import numpy as np

from ff_optimum.cores.optimizer.optimizer_base import Optimizer
from ff_optimum.cores.utilities import EventLogger

__all__ = ['SimulatedAnnealingBase']

logger = EventLogger(__name__)


class SimulatedAnnealingBase(Optimizer):

    """
    Base class for simulated annealing algorithm

    Attributes
    ----------
    _number_of_epoch
        number of epoch to be run for optimization

    __initial_temperature

    __final_temperature
        final temperature to be achived before terminating the optimization

    __cooling_rate
        rate for annealing the temperature

    __number_of_steps
        number of steps to be ran

    _acceptance_probability
        acceptance propability at the first step


    _threshold
        threshold for terminating the optimization

    _beta
        control parameter for controlling the acceptance propability

    Methods
    -------

    """

    __slots__ = ['_number_of_epoch', '__initial_temperature',
                 '__final_temperature', '__cooling_rate', '__number_of_steps',
                 '_number_of_stops', '_aceptance_ratio', '_threshold', '_beta',
                 '_changes_in_error']

    def __init__(self, number_of_processors: int, profile: str,
                 package_name: str, package_settings: dict,
                 parameters: Union[dict, np.ndarray],
                 constraints_source: str, constraints_input: Any,
                 command_holder_train: object, training_data: dict,
                 plot_information: dict, alogrithm_parameters: dict,
                 output_directory: str):

        self._number_of_epoch = int(alogrithm_parameters.get('epoch', 1))

        self.__initial_temperature = float(
            alogrithm_parameters.get('initial_temperature', 100))

        self.__final_temperature = float(
            alogrithm_parameters.get('final_temperature', 10))

        self.__cooling_rate = \
            float(alogrithm_parameters.get('cooling_rate', 0.85))

        self.__number_of_steps = \
            int(alogrithm_parameters.get('number_of_steps', 10))

        self._number_of_stops = \
            int(alogrithm_parameters.get('number_of_stops', 1))

        self._acceptance_probability = \
            float(alogrithm_parameters.get('acceptance_probability', 0.5))

        self._threshold = float(alogrithm_parameters.get('threshold', 0))

        self._beta = None

        self._changes_in_error = list()

        logger.info(f'Number of epoch {self._number_of_epoch}')
        logger.info(f'Initial temperature: {self.__initial_temperature:.4f}')
        logger.info(f'Final temperature: {self.__final_temperature:.4f}')
        logger.info(f'Cooling rate: {self.__cooling_rate:.4f}')
        logger.info(f'Number of steps: {self.__number_of_steps:.4f}')
        logger.info(f'Number of stops: {self._number_of_stops:.4f}')
        logger.info(f'Acceptance probability: '
                    f'{self._acceptance_probability:.4f}')
        logger.info(f'Threshold: {self._threshold:.4f}')

        super(SimulatedAnnealingBase, self).__init__(
            number_of_processors, profile, package_name, package_settings,
            parameters, constraints_source, constraints_input,
            command_holder_train, training_data, plot_information,
            output_directory)

    @abc.abstractclassmethod
    def _one_epoch(self):
        pass

    def _temperature_generator(self) -> Generator:
        """
        Generate temperature base on the initial temperature ,final temperature
        and the number of steps in the algorithm parameters

        Yields
        -----
        temperature
            the current temperature for the algorithm

        step
            the number of step for the algorithm
        """

        temperature = self.__initial_temperature

        while temperature > self.__final_temperature:

            for step in range(self.__number_of_steps):
                yield temperature, step

            temperature *= self.__cooling_rate

    def _metropolis_criteria(self, change_in_error: float,
                             temperature: float) -> bool:
        """
        determine whether the new solution is accepted by the metropholis
        criteria

        Parameters
        ----------
        change_in_error
            the difference of error between new solution and current solution
        temperature
            the current temperature in the simulated annealing algorithm

        Returns
        -------
        True
            If the change in error is smaller than zero or
            p_accept is greater than a random number
            between zero and one

        False
            Otherwise
        """

        if change_in_error <= 0:
            return True

        if self._beta is None:
            p_accept = self._acceptance_probability
        else:
            p_accept = min(
                np.exp(-1 * change_in_error / self._beta / temperature), 1)

        if (np.random.rand() < p_accept):
            return True

        return False

    def _find_beta(self, number_of_trial: int, changes_in_error: list) -> None:
        """
        determine the beta for the metropholis criteria with initial accpetance
        propability 0.5

        Parameters
        ----------
        changes_in_error
            the difference of error in a step
        number_of_trial
            the number of trial of a step

        """

        change_in_error = reduce(
            operator.add, changes_in_error) / number_of_trial

        self._beta = (-1 * change_in_error /
                      self.__initial_temperature /
                      np.log(self._acceptance_probability))

        logger.info(f'Beta: {self._beta}')
