# -*- coding: utf-8 -*-
from itertools import filterfalse
import re
from typing import Optional, Union

import numpy as np

from ff_optimum.cores.utilities.event_logging import EventLogger
from ff_optimum.cores.utilities.exceptions import FileEmptyError
from ff_optimum.cores.utilities.file_path import file_is_file_path_valid
from ff_optimum.user_packages.reaxff.optimization_setting import (
    REAXFF_PARAM_DTYPE, REAXFF_GENERAL_PARAMS, REAXFF_TYPE_PARAMS,
    REAXFF_BOND_PARAMS, REAXFF_DIAG_PARAMS, REAXFF_ANGLE_PARAMS,
    REAXFF_TORSION_PARAMS, REAXFF_HYDROGEN_BOND_PARAMS)

__all__ = ['read_reactive_force_field']

logger = EventLogger(__name__)


def read_reactive_force_field(force_field_directory: str) -> dict:
    """
    Read the reactive fore field parameters from file

    Parameters
    ----------
    force_field_directory
        file path of the force field

    Returns
    -------
    parameters

    Raises
    ------
    FileNotFoundError
        when the force field file is not found

    FileEmptyError
        when the force field file is empty

    KeyError
        when the key is not found

    ValueError
        when the number of parameters read actually is not equal to
        the one stated in the file

    See Also
    --------
    ReactiveForceFieldReader
    cores.utilities.exceptions.FileEmptyError

    """

    try:
        reader = ReactiveForceFieldReader(force_field_directory)
    except (FileNotFoundError, FileEmptyError, KeyError, ValueError) as e:

        logger.info(f'Read force field {force_field_directory} failed')

        raise e

    else:
        logger.info(f'Read force field {force_field_directory} sucess')

    return reader.parameters


class ReactiveForceFieldReader(object):

    """
    Class for reading ReaxFF parameters file
    """

    __TYPE_LINE_WIDTH = 4

    __BOND_LINE_WIDTH = 2

    __NUMBER_PATTERN = re.compile('(\d+)(\s+!\s+)([\w+\s\-*\w+]{1,})')

    __VALUES_PATTERN = re.compile('(-*\d+.\d+)')

    __GENERAL_PARAM_PATTERN = re.compile('(-*\d+.\d+)( !)(.+)')

    __TYPE_PATTERN = re.compile('(\s*[\w+]*\s+)(.+)')

    __BOND_PATTERN = re.compile('(\s*[\d+\s+\d+]*\s+)(.+)')

    __OFF_DIAGONAL_PATTERN = re.compile('(\s*\d+\s+\d+\s+)(.+)')

    __ANGLE_PATTERN = re.compile('(\s*\d+\s+\d+\s+\d+\s+)(.+)')

    __TORSION_PATTERN = re.compile('(\s*\d+\s+\d+\s+\d+\s+\d+\s+)(.+)')

    __HYDROGEN_BOND_PATTERN = re.compile('(\s*\d+\s+\d+\s+\d+\s+)(.+)')

    __slots__ = ['__parameters']

    def __init__(self, force_field_directory: Optional[str]=None) -> None:

        self.__parameters = None

        if force_field_directory:
            self.read_reactive_force_field(force_field_directory)

    @property
    def parameters(self):
        return self.__parameters

    def read_reactive_force_field(self,
                                  reactive_force_field_directory: str) -> None:
        """
        Read the reactive force field parameters from the file

        Parameters
        ----------
        reactive_force_field_directory
            file path of the force field file

        Returns
        -------
        None

        See Also
        --------
        ReactiveForceFieldReader.__read_general_parameters
        ReactiveForceFieldReader.__read_atoms
        ReactiveForceFieldReader.__read_bonds
        ReactiveForceFieldReader.__read_off_diagonal
        ReactiveForceFieldReader.__read_angle
        ReactiveForceFieldReader.__read_torsion
        ReactiveForceFieldReader.__read_hydrogen_bonds
        cores.utilities.file_path.file_is_file_path_valid
        """

        file_is_file_path_valid(reactive_force_field_directory)

        with open(reactive_force_field_directory, 'r') as fp:
            lines = list(map(lambda line: line.strip('\n'),
                             filterfalse(lambda line: line.startswith('#'),
                                         fp.readlines())))

        parameters = dict()

        idx = 0

        while idx < (len(lines)):

            res = re.match(self.__NUMBER_PATTERN, lines[idx].strip())

            if res:

                number_of_param = int(res.group(1))

                param_type = res.group(3).split(' ')[2]

                if number_of_param > 0:

                    if 'general' in param_type:

                        idx += 1

                        parameters[param_type] = \
                            self.__read_general_parameters(
                                lines[idx:(idx + number_of_param)])

                        idx += number_of_param

                    elif 'atoms' in param_type:

                        idx += self.__TYPE_LINE_WIDTH

                        elements, parameters[param_type] = self.__read_atoms(
                            lines[idx:(idx + number_of_param *
                                       self.__TYPE_LINE_WIDTH)],
                            number_of_param)

                        idx += number_of_param * self.__TYPE_LINE_WIDTH

                    elif 'bonds' in param_type:

                        idx += self.__BOND_LINE_WIDTH

                        parameters[param_type] = self.__read_bonds(
                            lines[idx:(idx + number_of_param *
                                       self.__BOND_LINE_WIDTH)],
                            elements,
                            number_of_param)

                        idx += number_of_param * self.__BOND_LINE_WIDTH

                    elif 'off-diagonal' in param_type:

                        idx += 1

                        parameters[param_type] = self.__read_off_diagonal(
                            lines[idx:(idx + number_of_param)],
                            elements, number_of_param)

                        idx += number_of_param

                    elif 'angles' in param_type:

                        idx += 1

                        parameters[param_type] = self.__read_angle(
                            lines[idx:(idx + number_of_param)],
                            elements, number_of_param)

                        idx += number_of_param

                    elif 'torsions' in param_type:

                        idx += 1

                        parameters[param_type] = self.__read_torsion(
                            lines[idx:(idx + number_of_param)],
                            elements, number_of_param)

                        idx += number_of_param

                    elif 'hydrogen' in param_type:

                        idx += 1

                        parameters[param_type] = self.__read_hydrogen_bonds(
                            lines[idx:(idx + number_of_param)],
                            elements, number_of_param)

                        idx += number_of_param

                else:
                    idx += 1

        self.__parameters = parameters

    def __read_general_parameters(self, lines: list) -> dict:
        """
        Read the general parameters from the force field file

        lines:
            lines containing general parameters

        Returns
        -------
        general_parameters

        Raises
        ------
        ValueError
            when number of general parameters read is not equal to
            number of REAXFF_GENERAL_PARAMS
        """

        general_parameters = dict.fromkeys(REAXFF_GENERAL_PARAMS, float)

        idx = 0

        for line in lines:

            res = re.match(self.__GENERAL_PARAM_PATTERN, line.strip())

            key = res.group(3)

            if not key in REAXFF_GENERAL_PARAMS:
                key = REAXFF_GENERAL_PARAMS[idx]

            general_parameters[key] = float(res.group(1))

            idx += 1

        if len(REAXFF_GENERAL_PARAMS) != len(general_parameters):
            raise ValueError('Number of general parameters mismatch')

        return general_parameters

    def __read_atoms(self, lines: list,
                     number_of_atoms: int) -> Union[list, dict]:
        """
        Read the atom parameters from the force field file

        lines:
            lines containing general parameters

        number_of_atoms:
            number of atoms stated in the file

        Returns
        -------
        elements
            name of the elements in the file
        type_parameters
            parameter values of the elements

        Raises
        ------
        ValueError
            when number of atoms parameters read is not equal to
            number of atoms stated in the file
        """

        type_parameters = dict()

        elements = ['*']

        idx = 0

        while idx < len(lines):

            values = []

            for i in range(self.__TYPE_LINE_WIDTH):

                res = re.match(self.__TYPE_PATTERN, lines[idx + i])

                element = res.group(1).strip()

                if element:

                    elements.append(element)

                    element_str = element

                values.extend(re.findall(self.__VALUES_PATTERN, res.group(2)))

            else:

                value_array = np.zeros(len(REAXFF_TYPE_PARAMS),
                                       dtype=REAXFF_PARAM_DTYPE)

                value_array['name'] = np.array(REAXFF_TYPE_PARAMS, dtype='U10')

                value_array['value'] = np.array(values, dtype='f8')

                type_parameters[element_str] = value_array

                idx += self.__TYPE_LINE_WIDTH

        if number_of_atoms != len(elements) - 1:
            raise ValueError('Nr of atoms mismatch')

        return elements, type_parameters

    def __read_bonds(self, lines: list, elements: list,
                     number_of_bonds: int) -> dict:
        """
        Read the bonds parameters from the force field file

        lines:
            lines containing general parameters

        elements
            name of the elements in the file

        number_of_bonds:
            number of bonds stated in the file

        Returns
        -------
        bonds_parameter
            bonds parameter values

        Raises
        ------
        ValueError
            when number of bonds read is not equal to
            number of bonds stated in the file

        See Also
        --------
        ReactiveForceFieldReader.__get_string
        """

        bonds_parameter = {}

        idx = 0

        while idx < len(lines):

            values = []

            for i in range(self.__BOND_LINE_WIDTH):

                res = re.match(self.__BOND_PATTERN, lines[idx])

                if res:

                    bond = (res.group(1).strip())

                    if bond:

                        bond_str = self.__get_string(bond, elements, 2)

                    values.extend(re.findall(self.__VALUES_PATTERN,
                                             res.group(2)))

                idx += 1

            else:

                bond_values = np.zeros(len(REAXFF_BOND_PARAMS),
                                       dtype=REAXFF_PARAM_DTYPE)

                bond_values['name'] = np.array(REAXFF_BOND_PARAMS, dtype='U10')

                bond_values['value'] = np.array(values, dtype=np.float)

                bonds_parameter[bond_str] = bond_values

        return bonds_parameter

    def __read_off_diagonal(self, lines: list, elements: list,
                            number_of_off_diagonal: int) -> dict:
        """
        Read the off-diagonal parameters from the force field file

        lines:
            lines containing general parameters

        elements
            name of the elements in the file

        number_of_off_diagonal:
            number of off-diagonal stated in the file

        Returns
        -------
        off_diagonal_parameters
            off_diagonal_parameters parameters values

        Raises
        ------
        ValueError
            when number of off-diagonal read is not equal to
            number of off-diagonal stated in the file

        See Also
        --------
        ReactiveForceFieldReader.__read_parameters
        """

        off_diagonal_parameters = self.__read_parameters(
            lines, self.__OFF_DIAGONAL_PATTERN, elements,
                REAXFF_DIAG_PARAMS, 2)

        if(number_of_off_diagonal != len(off_diagonal_parameters)):
            raise ValueError('Nr of off-diagonal terms mismatch')

        return off_diagonal_parameters

    def __read_angle(self, lines: list, elements: list,
                     number_of_angle: int) -> dict:
        """
        Read the angle parameters from the force field file

        lines:
            lines containing general parameters

        elements
            name of the elements in the file

        number_of_angle:
            number of angle stated in the file

        Returns
        -------
        angle_parameters
            angle parameters values

        Raises
        ------
        ValueError
            when number of angle read is not equal to
            number of angle stated in the file

        See Also
        --------
        ReactiveForceFieldReader.__read_parameters
        """

        angle_parameters = self.__read_parameters(
            lines, self.__ANGLE_PATTERN,
            elements, REAXFF_ANGLE_PARAMS, 3)

        if(number_of_angle != len(angle_parameters)):
            raise ValueError('Nr of angles mismatch')

        return angle_parameters

    def __read_torsion(self, lines: list, elements: list,
                       number_of_torsion: int) -> dict:
        """
        Read the torsion parameters from the force field file

        lines:
            lines containing general parameters

        elements
            name of the elements in the file

        number_of_torsion:
            number of torsion stated in the file

        Returns
        -------
        torsion_parameters
            torsion parameters values

        Raises
        ------
        ValueError
            when number of torsion read is not equal to
            number of torsion stated in the file

        See Also
        --------
        ReactiveForceFieldReader.__read_parameters
        """

        torsion_parameters = self.__read_parameters(
            lines, self.__TORSION_PATTERN,
            elements, REAXFF_TORSION_PARAMS, 4)

        if(number_of_torsion != len(torsion_parameters)):
            raise ValueError('Nr of torsions mismatch')

        return torsion_parameters

    def __read_hydrogen_bonds(self, lines: list, elements: list,
                              number_of_hygrogen_bond: int) -> dict:
        """
        Read the hydrogen bond parameters from the force field file

        lines:
            lines containing general parameters

        elements
            name of the elements in the file

        number_of_hygrogen_bond:
            number of hydrogen bond stated in the file

        Returns
        -------
        hydrogen_bond_parameters
            torsion parameters values

        Raises
        ------
        ValueError
            when number of hydrogen bond read is not equal to
            number of hydrogen bond stated in the file

        See Also
        --------
        ReactiveForceFieldReader.__read_parameters
        """

        hydrogen_bond_parameters = self.__read_parameters(
            lines, self.__HYDROGEN_BOND_PATTERN, elements,
            REAXFF_HYDROGEN_BOND_PARAMS, 3)

        if number_of_hygrogen_bond != len(hydrogen_bond_parameters):
            raise ValueError('Nr of hydrogen bonds mismatch')

        return hydrogen_bond_parameters

    @staticmethod
    def __read_parameters(lines: list, pattern: object,
                          elements: list,
                          param_names: list,
                          number_of_elements: int) -> dict:
        """
        Read parameters from the force field file

        Parameters
        ----------
        lines
            the lines containing parameters values
        pattern
            pattern of the specific type of parameters

        elements
            list of string of elements

        param_names
            name of the parameters

        number_of_elements
            number of elements in the specific type


        See Also
        --------
        ReactiveForceFieldReader.__get_string
        """

        parameters = {}

        for line in lines:

            res = re.match(pattern, line)

            if res:

                parameter_str = ReactiveForceFieldReader.__get_string(
                    res.group(1).strip(), elements, number_of_elements)

                values = np.zeros(len(param_names), dtype=REAXFF_PARAM_DTYPE)

                values['name'] = np.array(param_names, dtype='U10')

                values['value'] = np.array(
                    re.findall(ReactiveForceFieldReader.__VALUES_PATTERN,
                               res.group(2)), dtype=np.float)

                parameters[parameter_str] = values

        return parameters

    @staticmethod
    def __get_string(something: str, elements: list,
                     number_of_elements: int) -> str:
        """
        Get the bond string from index to elements

        e.g. 1 2 3 and the elements is H S Ge then the output will be H-S-Ge

        Parameters
        ----------
        something
            element in index

        elements
            element in string

        number_of_elements
            number of elements in the string

        Returns
        -------
        bond string in elements
        """

        res = str()

        atoms = list(map(int, something.split('  ')))

        for idx in range(number_of_elements - 1):
            res += elements[atoms[idx]] + '-'
        else:
            res += elements[atoms[-1]]

        return res
