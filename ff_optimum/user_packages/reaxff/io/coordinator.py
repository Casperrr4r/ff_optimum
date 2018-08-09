# -*- coding: utf-8 -*-
import copy
import os
import platform

from .compile_commands import compile_lammps_commands
from .read_training_data import DftXmlReader
from ff_optimum.cores.utilities import (
    argument_type_check, file_get_filenames_from_directory,
    file_set_path, file_is_directory_valid)

__all__ = ['DftXmlCoordinator']


@argument_type_check
def _read_and_compile_dft_xml(xml_path: str, reaxff_path: str) -> tuple:
    """
    Read and compile DFT training data to LAMMPS commands and Training data

    Parameters
    ----------
    xml_path
        path of the xml training data

    reaxff_path
        path of the ReaxFF parameters

    Returns
    -------
    name of the molecule

    information for visulization

    compile_commands

    training data
    """

    with DftXmlReader() as reader:

        reader.read_xml(xml_path)

        res = reader.results

        compiled_commands = (
            compile_lammps_commands(res, reader.flags, reaxff_path))

        plot_info = {'scantype': res['scantype'],
                     'values': reader.plot_values}

        return (res['name'], plot_info, compiled_commands,
                reader.training_data)


class DftXmlCoordinator(object):

    """
    Class for reading all of the xml config

    Attributes
    ----------
    __compiled_commands
    __training_datasets
    __plot_informations

    Methods
    -------
    read_and_compile_configs(config_directory, temp_directory)
    """

    __slots__ = ['__compiled_commands', '__training_datasets',
                 '__plot_informations']

    def __init__(self) -> None:

        self.__compiled_commands = {}

        self.__training_datasets = {}

        self.__plot_informations = {}

    def __enter__(self) -> object:
        return self

    def __exit__(self, exc_ty, exc_val, tb) -> None:
        pass

    @property
    def compiled_commands(self) -> dict:
        return copy.deepcopy(self.__compiled_commands)

    @property
    def training_datasets(self) -> dict:
        return copy.deepcopy(self.__training_datasets)

    @property
    def plot_informations(self) -> dict:
        return copy.deepcopy(self.__plot_informations)

    def read_and_compile_configs(self, config_directory: str,
                                 temp_directory: str=os.getcwd()) -> None:
        """
        Read and compile the xml training datasets

        Parameters
        ---------
        config_directory
            directory containing all the xmls to be read

        temp_directory
            temporary directory

        Returns
        -------
        None

        See Also
        --------
        _read_and_compile_dft_xml
        """

        reaxff_path = file_set_path(temp_directory, 'ffield_temp')

        xml_paths = ([config_directory] if config_directory.endswith('.xml')
                     else file_get_filenames_from_directory(config_directory,
                                                            'xml'))

        for xml_path in xml_paths:

            res = _read_and_compile_dft_xml(xml_path, reaxff_path)

            mol_name = res[0].lower()

            self.__training_datasets[mol_name] = res[3]

            self.__compiled_commands[mol_name] = res[2]

            self.__plot_informations[mol_name] = res[1]
