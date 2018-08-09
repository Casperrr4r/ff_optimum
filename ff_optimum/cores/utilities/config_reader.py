# -*- coding: utf-8 -*-
import importlib
import os
from typing import Any, Optional

from ff_optimum.cores.utilities.argument_type_check import argument_type_check
from ff_optimum.cores.utilities.command_holder import CommandsHolder
from ff_optimum.cores.utilities.event_logging import EventLogger
from ff_optimum.cores.utilities.exceptions import (
    FileEmptyError, XmlNodeNotFoundError)
from ff_optimum.cores.utilities import file_path as fs

logger = EventLogger(__name__)


class ConfigReader(object):

    __slots__ = ['__number_of_processors', '__profile', '__package_name',
                 '__package_settings', '__directory', '__temp_directory',
                 '__param_initial_values',
                 '__constraints_source', '__constraints_input',
                 '__commands_holder_train', '__training_data',
                 '__plot_information', '__algorithm_name',
                 '__alogrithm_parameters', '__output_directory']

    """
    Class for reading json config file

    Attributes
    ----------
    number_of_processors: int
        The number of processors to be used

    profile: str
        The profile name for running ipyparallel

    package_name: str
        The name of the package to be used during optimization

    package_settings: dict
        The dictionary object containing specific package setting

    directory: str
        The string storing directory for reading input files

    temp_directory: str
        The string storing temp directory for optimization

    param_initial_values: dict
        The dictionary object containing parameters initial values

    constraints_source: str
        Type of constraint source

    constraints_input: Any
        Input of constraints

    commands_holder_train: CommandsHolder

    training_data: dict
        The dictionary object containing training data

    algorithm_name: str
        The name of the algorithm to be used for optimization

    alogrithm_parameters
        The dictionary object containing alogithm parameters

    output_directory: str
        The string storing directory for writing outputs

    Methods
    -------
    read_config(config_path)
        read the config file specific in the config path
    """

    def __init__(self, *args: str) -> None:

        self.__number_of_processors = None
        self.__profile = None

        self.__package_name = None
        self.__package_settings = None

        self.__directory = None
        self.__temp_directory = None

        self.__param_initial_values = None

        self.__constraints_source = 'default'
        self.__constraints_input = None

        self.__commands_holder_train = None
        self.__training_data = None
        self.__plot_information = None

        self.__algorithm_name = None
        self.__alogrithm_parameters = None

        self.__output_directory = None

        if args is not None:
            self.read_config(args[0])

    @property
    def number_of_processors(self) -> int:
        return self.__number_of_processors

    @property
    def profile(self) -> str:
        return self.__profile

    @property
    def package_name(self) -> str:
        return self.__package_name

    @property
    def package_settings(self) -> dict:
        return self.__package_settings

    @property
    def param_initial_values(self) -> dict:
        return self.__param_initial_values

    @property
    def constraints_source(self) -> str:
        return self.__constraints_source

    @property
    def constraints_input(self) -> Any:
        return self.__constraints_input

    @property
    def commands_holder_train(self) -> CommandsHolder:
        return self.__commands_holder_train

    @property
    def training_data(self) -> dict:
        return self.__training_data

    @property
    def plot_information(self) -> dict:
        return self.__plot_information

    @property
    def algorithm_name(self) -> str:
        return self.__algorithm_name

    @property
    def alogrithm_parameters(self) -> dict:
        return self.__alogrithm_parameters

    @property
    def output_directory(self) -> str:
        return self.__output_directory

    @argument_type_check
    def read_config(self, config_path: str) -> None:
        """
        Function for reading json config file

        Parameters
        ----------
        config_path
            The string object containing config file path

        Returns
        -------
        None

        Raises
        ------
        KeyError
            when key is not found in the config
        TypeError
            when the input type is different with type annotatio

        IsADirectoryError
            when the input file is a directory not a file

        XmlNodeNotFoundError
            when the xml node is not found

        FileEmptyError
            when the file is empty

        FileNotFoundError
            when the file is not found

        NotImplementedError
            when the function is not implemented

        See Also
        --------
        ConfigReader.__read_settings
        cores.utilities.file_path.file_get_filename_from_path
        cores.utilities.file_path.file_read_json
        """

        try:

            filename = fs.file_get_filename_from_path(config_path)

            logger.info(f'Start reading config {filename}')

            self.__read_settings(fs.file_read_json(config_path))

        except (KeyError, TypeError, ValueError, IsADirectoryError,
                XmlNodeNotFoundError, FileEmptyError, FileNotFoundError,
                NotImplementedError) as e:

            logger.error(e)

            logger.error(f'Read config {filename} failed')

            raise

        else:
            logger.info(f'Read config {filename} sucess')

    def __read_settings(self, setting: dict) -> None:
        """
        Read settings

        Parameters
        ----------
        setting
            settings from config file

        Returns
        -------
        None

        See Also
        --------
        ConfigReader.__get_contents_from_json_config
        ConfigReader.__read_processors_setting
        ConfigReader.__read_package_setting
        ConfigReader.__read_input_setting
        ConfigReader.__read_parameters_setting
        ConfigReader.__read_param_from_file
        ConfigReader.__read_training_setting
        """

        self.__read_processors_setting(
            setting.get('processors_setting', None))

        self.__directory = setting.get('directory', None)

        self.__temp_directory = setting.get('directory', os.getcwd())

        self.__read_package_setting(setting['package'])

        self.__read_input_setting(setting['input'])

        self.__read_alogrithm_setting(setting['alogrithm'])

        self.__read_output_setting(setting['output'])

    def __read_processors_setting(self, setting: Optional[dict]) -> None:
        """
        Read processors setting

        Parameters
        ----------
        setting
            Dictionary object containing all the setting

        Returns
        -------
        None
        """

        logger.info(f'Start reading processors setting')

        if setting is None:

            self.__number_of_processors = 1

            self.__profile = None

        else:

            self.__number_of_processors = setting['number_of_processors']

            self.__profile = setting.get('profile', None)

        if self.__number_of_processors > 1 and self.__profile is None:
            raise ValueError('Profile is missing please check the config')

        logger.info(f'Read processor setting sucesses')

        logger.info(f'number of processors: {self.__number_of_processors}, '
                    f'profile: {self.__profile}')

    def __read_package_setting(self, setting: dict) -> None:
        """
        Read package setting

        Parameters
        ----------
        setting
            Dictionary object containing all the setting

        Returns
        -------
        None

        See Also
        --------
        importlib.import_module
        """

        logger.info(f'Start reading package setting')

        self.__package_name = setting.pop('name')

        package = importlib.import_module(f'ff_optimum.user_packages.'
                                          f'{self.__package_name}')

        logger.info(f'Package name: {self.__package_name}')

        self.__package_settings = package.read_package_setting(setting)

    def __read_input_setting(self, setting: dict) -> dict:
        """
        Read input setting

        Parameters
        ----------
        setting
            Dictionary object containing all the setting

        Returns
        -------
        None

        See Also
        --------
        ConfigReader.__read_parameters_setting
        ConfigReader.__read_training_setting
        """

        logger.info(f'Start reading input setting')

        self.__read_parameters_setting(setting['parameters'])

        self.__read_training_setting(setting['training_data'])

    def __read_parameters_setting(self, parameter_setting: dict) -> None:
        """
        Read parameters setting

        Parameters
        ----------
        setting
            Dictionary object containing parameters setting

        Returns
        -------
        dict
            Dictionary object containing parameters initial values

        See Also
        --------
        cores.utilities.file_path.file_set_path
        importlib.import_module
        """

        if self.__directory is not None:
            param_path = fs.file_set_path(
                self.__directory, parameter_setting['input'])
        else:
            param_path = parameter_setting['input']

        package = importlib.import_module(f'ff_optimum.user_packages.'
                                          f'{self.__package_name}')

        self.__param_initial_values = (
            package.read_parameters_from_file(param_path))

    def __read_training_setting(self, training_setting: dict) -> None:
        """
        Read training data setting

        Parameters
        ----------
        setting
            Dictionary object containing training data setting

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            if the file type is not xml

        See Also
        --------
        ConfigReader__read_training_data_from_file
        """

        if 'xml' in training_setting['type']:
            if self.__directory is not None:
                path = fs.file_set_path(self.__directory,
                                        training_setting['input'])
            else:
                path = training_setting['input']

            res = self.__read_training_data_from_file(
                training_setting['mode'], path)

            self.__commands_holder_train = res[0]

            self.__training_data = res[1]

            self.__plot_information = res[2]

        else:
            raise NotImplementedError()

    def __read_training_data_from_file(self, mode: str, path: str) -> tuple:
        """
        Read training data from file

        Parameters
        ----------
        mode
            Single file mode or mult-file mode

        path
            File path of the training datas

        Returns
        -------
        holder
            CommandsHolder object storing the compiled commands

        dict
            Dictionary object containing training data

        See Also
        --------
        cores.utilities.CommandsHolder

        """

        holder = CommandsHolder()

        package = importlib.import_module(f'ff_optimum.user_packages.'
                                          f'{self.__package_name}')

        coordinator = package.Coordinator()

        coordinator.read_and_compile_configs(path, self.__temp_directory)

        holder.compiled_commands = coordinator.compiled_commands

        holder.excutor = getattr(package.compute, 'compute_values')

        holder.temp_directory = self.__temp_directory

        training_data = coordinator.training_datasets

        plot_informations = coordinator.plot_informations

        return (holder, training_data, plot_informations)

    def __read_alogrithm_setting(self, algorithm_setting: dict) -> None:
        """
        Read algorithn setting

        Parameters
        ----------
        algorithm_setting
            Dictionary object containing algorithm setting

        Returns
        -------
        dict
            Dictionary object containing training data
        """

        logger.info(f'Start reading algorithm setting')

        self.__alogrithm_parameters = algorithm_setting

        self.__algorithm_name = self.__alogrithm_parameters.pop('name')

        logger.info(f'Read algorithm setting sucess')

        logger.info(f'Alogithm: {self.__algorithm_name}')

    def __read_output_setting(self, setting: Optional[dict]) -> None:
        """
        Read output setting

        Parameters
        ----------
        setting
            Dictionary object containing output setting

        Returns
        -------
        None

        See Also
        --------
        cores.utilities.file_path.file_set_path
        """

        logger.info(f'Start reading output setting')

        directory = (os.getcwd() if setting is None else
                     setting.get('directory', os.getcwd()))

        if self.__directory is not None:
            directory = fs.file_set_path(self.__directory, directory)

        self.__output_directory = directory

        logger.info(f'Read output setting sucess')

        logger.info(f'Output directory: {self.__output_directory}')
