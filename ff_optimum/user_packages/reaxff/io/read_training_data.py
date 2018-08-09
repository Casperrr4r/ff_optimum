# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
from typing import Optional, Union

import numpy as np

from ff_optimum.user_packages.reaxff.optimization_setting.parameters import (
    REAXFF_NUMBER_OF_ENERGY, REAXFF_NUMBER_OF_STRESS)

from ff_optimum.user_packages.reaxff.compute.compute_angles_distances_volumes \
    import *

from ff_optimum.cores.utilities import EventLogger, FileEmptyError
import ff_optimum.cores.utilities.xml_parse as xml_parse

logger = EventLogger(__name__)

__all__ = ['DftXmlReader']


class DftXmlReader(object):

    """
    Class for reading DFT xml training data

    Attributes
    ----------
    __flags: dict
    __parse_results: dict
    __training_data: dict
    __plot_values: list


    Methods
    -------
    read_xml(xml_path)
        read the DFT xml

    """

    __XML_KEYS = ['NAME', 'CODE', 'UNIT', 'FUNC', 'BASIS',
                  'SPIN', 'CHARGE', 'SCANTYPE', 'MASS',
                  'PERIODICITY', 'TYPES', 'BOND']

    __BOND_SCANTYPE = ['DISS', 'ANG', 'TOR']

    __POSITION_KEYS = ['BOX', 'TYPE', 'X', 'Y', 'Z']

    __TRAINING_DATA_KEYS = ['Q', 'FX', 'FY', 'FZ', 'STRESS', 'ENERGY']

    __slots__ = ['__flags', '__parse_results',
                 '__training_data', '__plot_values']

    def __init__(self, xml_path: Optional[str]=None):

        self.__flags = {'energy': True, 'q': True,
                        'fx': True, 'fy': True, 'fz': True,
                        'stress': True}

        self.__parse_results = None

        self.__training_data = None

        if xml_path:
            self.read_xml(xml_path)

    @property
    def flags(self) -> dict:
        return self.__flags

    @property
    def results(self) -> dict:
        return self.__parse_results

    @property
    def training_data(self) -> list:
        return self.__training_data

    @property
    def plot_values(self) -> list:
        return self.__plot_values

    def __enter__(self) -> object:
        return self

    def __exit__(self, exc_ty, exc_val, tb) -> None:
        pass

    def read_xml(self, xml_path: str) -> None:
        """
        Read the DFT xml training data

        Parameters
        ----------
        xml_path
            file path of the xml training data file

        Reutrns
        -------
        None

        Raises
        ------
        TypeError
            raise when the input xml file path is not a string object

        FileNotFoundError
            raise when the fle specify in the xml path is not found

        FileEmptyError
            raise when the fle specify in the xml path is empty

        See Also
        --------
        cores.utilities.xml_parse.get_text_from_node
        DftXmlReader.__read_step

        """

        result = {}

        training_dataset = []

        try:
            root = xml_parse.get_root_from_file(xml_path)

            for key in self.__XML_KEYS[:-1]:
                result[key.lower()] = xml_parse.get_text_from_node(root, key)

            result['types'] = np.array(result['types'].split(), dtype=np.int)

            result['positions'] = []

            for (position, training_data) in map(self.__read_step,
                                                 root.iter('STEP')):

                natom = position['type'].size

                result['positions'].append(position)

                training_dataset.append(training_data)

            values = []

            if result['scantype'] in self.__BOND_SCANTYPE:

                bond_indices = np.array(
                    xml_parse.get_text_from_node(root, 'BOND').split(' '),
                    dtype=np.int)

                bond_indices -= 1

                for position in result['positions']:

                    positions = [self.__get_position_by_index(
                        position, bond_index) for bond_index in bond_indices]

                    if bond_indices.size == 2:
                        values.append(
                            compute_bond_distance(positions[0], positions[1]))
                    elif bond_indices.size == 3:
                        values.append(compute_bond_angle(
                            positions[0], positions[1], position[2]))
                    else:
                        values.append(
                            compute_torsion_angle(positions[0], positions[1],
                                                  position[2], positions[3]))
            elif 'EOS' in result['scantype']:
                for position in result['positions']:
                    values.append(compute_volume(position['box']) / natom)

            else:
                values.extend(list(range(len(position['x'][0]))))

            self.__parse_results = result

            self.__plot_values = values

            self.__training_data = training_dataset

        except (TypeError, FileNotFoundError, FileEmptyError) as e:

            logger.info('Read DFT training data %s failed' % xml_path)

            logger.error(e)

            raise e

        else:
            logger.info('Read DFT training data %s sucess' % xml_path)

    @staticmethod
    def __get_position_by_index(position: dict, index: int):
        return np.array([position['x'][0][index], position['y'][0][index],
                         position['z'][0][index]], dtype='f8')

    def __read_step(self, xml_step: ET.Element) -> Union[dict, dict]:
        """
        Read the data inside step node

        Parameters
        ----------
        xml_step
            the xml step node

        Returns
        -------
        positions
            positions of the atoms

        training_data
            the training data of the step

        See Also
        --------
        cores.utilities.xml_parse.get_text_from_node
        DftXmlReader.__read_flag_and_text_from_node
        """

        positions, training_data = {}, {}

        box = xml_parse.get_text_from_node(xml_step, 'BOX')

        positions['box'] = \
            np.array(box.split(' '), dtype=np.float128) if box else None

        positions['type'] = np.array(
            xml_parse.get_text_from_node(xml_step, 'TYPE').split(' '),
            dtype=np.int)

        number_of_atoms = positions['type'].size

        lengths = [number_of_atoms] * 4

        lengths.extend([REAXFF_NUMBER_OF_STRESS, REAXFF_NUMBER_OF_ENERGY])

        for key in self.__POSITION_KEYS[2:]:
            positions[key.lower()] = self.__read_flag_and_text_from_node(
                xml_step, key, positions['type'].size)

        for key, length in zip(self.__TRAINING_DATA_KEYS, lengths):
            training_data[key.lower()], self.__flags[key.lower()] = \
                self.__read_flag_and_text_from_node(xml_step, key, length)

        training_data['energy'] /= number_of_atoms

        return positions, training_data

    @staticmethod
    def __read_flag_and_text_from_node(
            xml_node: ET.Element,
            attribute: str,
            length: int) -> Union[float, np.ndarray]:
        """
        Read flag and texts from the xml_node

        Returns
        -------
        value: float
            value in the node

        flag: bool
            flag in the node

        See Also
        --------
        cores.utilities.xml_parse.get_flag_from_node
        cores.utilities.xml_parse.get_text_from_node
        """

        flag = xml_parse.get_flag_from_node(xml_node, attribute)

        if 'ENERGY' in attribute:
            value = (float(xml_parse.get_text_from_node(xml_node, attribute))
                     if flag else 0)
        else:
            value = (np.array(xml_parse.get_text_from_node(
                xml_node, attribute).split(' '), dtype=np.float)
                if flag else np.zeros(length, dtype=np.float))

        return value, flag
