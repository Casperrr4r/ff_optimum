# -*- coding: utf-8 -*-
import abc
import copy
import importlib
from threading import Condition, Thread
from typing import Any, Generator, Optional, Union

import numpy as np

from ff_optimum.cores.utilities import (EventLogger, file_set_path,
                                        start_client, wait_until_client_ready)

logger = EventLogger(__name__)

__all__ = ['Optimizer']


class Optimizer(abc.ABC):

    """
    Abstarct Base class for optimizers

    Attributes
    ----------
    _package_name: str
        Name of the package

    _package: module
        Imported package

    _step_size: dict
        Dictionary containing step size for moving the parameters
        during optimization

    _package_setting_train: dict
        Dictionary object containing package setting for training

    __package_setting_test: dict
        Dictionary object containing package setting for testing

    __initial_parameters: dict
        Dictionary object containing initial parameters value

    _parameters: dict
        Dictionary object containing current parameters value

    __initial_fitness: object
        Object containing initial fitness value

    _constraints: dict
        Dictionary object containing the constraints

    _commands_holder_train: CommandHolder
        CommandHolder object storing the compiled command for training

    _training_dataset: dict
        Dictionary object containing training data

    __commands_holder_test: CommandHolder
        CommandHolder object storing the compiled command for testing

    __test_dataset: dict
        Dictionary object containing test data

    _output_directory: str
        String object containing output directory

    Methods
    -------
    optimize
    test
    save_parameters_to_file
    save_error_to_file
    """

    __slots__ = ['_package_name', '_package', '_step_size',
                 '_package_setting_train', '__package_setting_test',
                 '__initial_parameters', '_parameters', '__initial_fitness',
                 '_constraints', '_commands_holder_train', '_training_dataset',
                 '__plot_information', '__commands_holder_test',
                 '__test_dataset', '_output_directory']

    def __init__(self, number_of_processors: int, profile: str,
                 package_name: str, package_settings: dict,
                 parameters: Union[dict, np.ndarray],
                 constraints_source: str, constraints_input: Any,
                 command_holder_train: object, training_data: dict,
                 plot_information: dict, output_directory: str):

        self.__enable_parallel_client(number_of_processors, profile)

        logger.info(f'Package: {package_name}')

        self._package_name = package_name

        self._package = importlib.import_module(f'ff_optimum.user_packages.'
                                                f'{package_name.lower()}')

        self._step_size = self.__get_step_size(
            package_settings.pop("level"))

        self._package_setting_train = copy.deepcopy(package_settings)

        self._pop_unsued_training_objectives()

        self.__package_setting_test = (
            self.__perapre_test_setting(package_settings))

        self.__initial_parameters = copy.deepcopy(parameters)

        self._parameters = copy.deepcopy(parameters)

        self.__initial_fitness = None

        self._constraints = self._package.get_default_constraints()

        self._commands_holder_train = copy.deepcopy(command_holder_train)

        self._training_dataset = copy.deepcopy(training_data)

        self.__plot_information = copy.deepcopy(plot_information)

        self.__commands_holder_test = copy.deepcopy(command_holder_train)

        self.__test_dataset = copy.deepcopy(training_data)

        self._output_directory = output_directory

    @staticmethod
    def __enable_parallel_client(
            number_of_processors: int, profile: Optional[str]) -> None:
        """
        Create 2 thread, start_thread start the client and
        wait_thread wait for the client to be initialized

        Arguments
        ---------
        number_of_processors
            number of processors to be used

        profile
            name of the profile

        Returns
        -------
        None

        """

        logger.info(f'Number of processors: {number_of_processors}'
                    f', profile: {profile}')

        if number_of_processors > 1 and profile is not None:

            condition_varibale = Condition()

            start_thread = Thread(name='start', target=start_client,
                                  args=(condition_varibale,
                                        number_of_processors, profile))

            wait_thread = Thread(name='wait', target=wait_until_client_ready,
                                 args=(condition_varibale,))

            wait_thread.start()

            start_thread.start()

            wait_thread.join()

    @staticmethod
    def __perapre_test_setting(package_settings: dict) -> dict:
        """

        Arguments
        ---------
        package_settings
            Dictionary object containing package settings

        Returns
        -------
        __package_setting
            package_settings for testing

        """

        __package_setting = copy.deepcopy(package_settings)

        __package_setting['slient'] = False

        if __package_setting.get('weights') is not None:
            __package_setting['weights'] = [1, 1, 1, 1, 1]

        return __package_setting

    @property
    def initial_parameters(self) -> Optional[Union[float, object]]:
        return self.__initial_parameters

    @property
    def initial_fitness(self) -> Optional[Union[float, object]]:
        if self.__initial_fitness is None:
            self.__calculated__initial_fitness()

        return self.__initial_fitness

    @abc.abstractclassmethod
    def optimize(self) -> Any:
        pass

    @abc.abstractclassmethod
    def save_parameters_to_file(self, filename: str='ffield_out') -> Any:
        pass

    @abc.abstractclassmethod
    def save_error_to_file(self) -> Any:
        pass

    def test(self, visualize: bool=True) -> Any:
        """
        Test
        """

        logger.info('Start testing')

        test_calculated_values = (
            self.__commands_holder_test.execute_commands(self._parameters))

        logger.info('Start calculating test errors')

        test_errors = (self._evaluate_fitness(
            test_calculated_values, self.__test_dataset,
            self.__package_setting_test))

        if visualize:
            self._package.plot_fitness(
                self.__package_setting_test, self.__plot_information,
                test_calculated_values, self.__test_dataset,
                self._output_directory)

        logger.info('Testing finishes')

        return test_errors

    def __calculated_initial_fitness(self) -> None:
        """
        Evaluate the initial fitness

        Returns
        -------
        None

        """

        logger.info('Start calculating initial fitness')

        __calculated_values = (self._commands_holder_train.execute_commands(
            self.__initial_parameters))

        self.__initial_fitness = (self._evaluate_fitness(
            __calculated_values, self._training_dataset))

        logger.info('Initial fitness calculation finishes')

    def __get_step_size(self, inputs):
        return self._package.get_default_step_size(inputs)

    def _evaluate_fitness(self, calculated_values: dict,
                          training_datas: dict,
                          settings: Optional[dict]=None) -> Any:
        """
        Evaluate fitness
        """

        if not settings:
            settings = self._package_setting_train

        return (self._package.compute_errors(
            calculated_values, training_datas, settings))

    def _pop_unsued_training_objectives(self) -> None:
        self._package.pop_unused_training_objectives(
            self._package_setting_train)
