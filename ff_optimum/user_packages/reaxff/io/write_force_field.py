# -*- coding: utf-8 -*-
import os
from typing import Optional

import numpy as np

from ff_optimum.cores.utilities.file_path import file_create_directory
from ff_optimum.user_packages.reaxff.optimization_setting \
    import REAXFF_GENERAL_PARAMS

__all__ = ['save_reactive_force_field']


class ReactiveForceFieldWriter(object):

    @classmethod
    def write_force_field_to_file(cls, parameters: dict,
                                  directory: str) -> None:
        """
        Write the force field parameters to the file

        Parameters
        ----------
        parameters
            force field parameters

        directory
            target directory

        Returns
        -------
        None

        See Also
        --------
        ReactiveForceFieldWriter.__prepare_headers
        ReactiveForceFieldWriter.__prepare_general_parameters_contents
        ReactiveForceFieldWriter.__prepare_atoms_contents
        ReactiveForceFieldWriter.__prepare_bonds_contents
        ReactiveForceFieldWriter.__prepare_off_diagonals_contents
        ReactiveForceFieldWriter.__prepare_angles_contents
        ReactiveForceFieldWriter.__prepare_torsions_contents
        ReactiveForceFieldWriter.__prepare_hydrogen_bonds_contents
        """

        np.set_printoptions(formatter={'float_kind': lambda x: '%8.4f' % x})

        elements = parameters['atoms'].keys()

        with open(directory, 'w') as fp:

            fp.write(cls.__prepare_headers(elements))

            fp.writelines(
                cls.__prepare_general_parameters_contents(
                    parameters['general']))

            fp.writelines(cls.__prepare_atoms_contents(parameters['atoms']))

            fp.writelines(cls.__prepare_bonds_contents(
                parameters.get('bonds', None), elements))

            fp.writelines(
                cls.__prepare_off_diagonals_contents(
                    parameters.get('off-diagonal', None), elements))

            fp.writelines(
                cls.__prepare_angles_contents(
                    parameters.get('angles', None), elements))

            fp.writelines(
                cls.__prepare_torsions_contents(
                    parameters.get('torsions', None), elements))

            fp.writelines(
                cls.__prepare_hydrogen_bonds_contents(
                    parameters.get('hydrogen', None), elements))

    @staticmethod
    def __prepare_headers(elements: list) -> str:
        """
        Prepare the header of the force field file

        Parameters
        ----------
        elements
            elements in the force field

        Returns
        -------
        header

        """
        return f'# ReaxFF_{"-".join(elements)} optimized with ff_optimum\n'

    @staticmethod
    def __prepare_general_parameters_contents(general_params: dict) -> list:
        """
        Prepare the general parameters contnet of the force field file

        Parameters
        ----------
        general_params
            general parameters

        Returns
        -------
        general_parameters_contents
            general parameters content of the force field file
        """

        general_parameters_contents = list()

        general_parameters_contents.append(
            f' {len(REAXFF_GENERAL_PARAMS)} ! Number of general parameters\n')

        for key, value in general_params.items():
            general_parameters_contents.append(f'   {value:.4f} ! {key}\n')

        return general_parameters_contents

    @staticmethod
    def __prepare_atoms_contents(type_parameters: dict) -> list:
        """
        Prepare the atom parameters contnet of the force field file

        Parameters
        ----------
        type_parameters
            atom parameters

        Returns
        -------
        type_contents
            atom parameters content of the force field file
        """

        type_contents = []

        type_contents.append(
            ' {:^2}'.format(len(type_parameters)) +
            ' ! Nr of atoms; '
            'cov.r; valency;a.m;Rvdw;Evdw;gammaEEM;cov.r2;#el\n'
            '\t\t'
            'alfa;gammavdW;valency;Eunder;Eover;chiEEM;etaEEM;n.u.'
            '\n\t\t'
            'cov r3;Elp;Heat inc.;n.u.;n.u.;n.u.;n.u.\n'
            '\t\tov/un;val1;n.u.;val3,vval4\n')

        for element, values in type_parameters.items():

            for i in range(4):

                if i == 0:
                    content = ' ' * 2 + element + ' ' * (6 - len(element))
                else:
                    content = ' ' * 8

                content += \
                    (str(values['value'][8 * i: 8 *
                         (i + 1)]).lstrip('[').rstrip(']') + '\n')

                type_contents.append(content)

        return type_contents

    @staticmethod
    def __prepare_bonds_contents(bonds: Optional[list],
                                 elements: list) -> list:
        """
        Prepare the bonds parameters contnet of the force field file

        Parameters
        ----------
        bonds
            bonds parameters

        elements
            elements in the force field

        Returns
        -------
        bonds_content
            bond parameters content of the force field file

        See Also
        --------
        ReactiveForceFieldWriter.__get_num_from_str

        """

        bonds_content = []

        number_of_bond = len(bonds) if bonds is not None else 0

        bonds_content.append(
            ' {:^2}'.format(number_of_bond) +
                    ' ! Nr of bonds; Edis1;LPpen;n.u.;pbe1;pbo5;13corr;pbo6' +
                    '\n\t   pbe2;pbo3;pbo4;n.u.;pbo1;pbo2;ovcorr\n')

        if number_of_bond:

            for key, values in bonds.items():

                num = \
                    ReactiveForceFieldWriter.__get_num_from_str(elements, key)

                bonds_content.append(
                    '  ' + num + ' ' * 2 +
                            str(values['value'][:8]).lstrip('[').rstrip(']') +
                            '\n')

                val = str(values['value'][8:]).lstrip('[').rstrip(']')

                bonds_content.append('\t' + val + '\n')

        return bonds_content

    @staticmethod
    def __prepare_off_diagonals_contents(off_diagonals: Optional[dict],
                                         elements: list) -> list:
        """
        Prepare the off diagonals parameters contnet of the force field file

        Parameters
        ----------
        off_diagonals
            off diagonals parameters

        elements
            elements in the force field

        Returns
        -------
        off_diagonal_contents
            off diagonals parameters content of the force field file

        See Also
        --------
        ReactiveForceFieldWriter.__get_num_from_str
        """

        off_diagonal_contents = []

        number_of_off_diagonal = \
            len(off_diagonals) if off_diagonals is not None else 0

        off_diagonal_contents.append(
            ' {:^2}'.format(number_of_off_diagonal) +
                    ' ! Nr of off-diagonal terms; ' +
                    'Ediss;Ro;gamma;rsigma;rpi;rpi2\n')

        if number_of_off_diagonal:

            for key, values in off_diagonals.items():

                num = \
                    ReactiveForceFieldWriter.__get_num_from_str(elements, key)

                off_diagonal_contents.append(
                    ' ' * 2 + num + ' ' * 2 +
                            str(values['value']).lstrip('[').rstrip(']') +
                            '\n')

        return off_diagonal_contents

    @staticmethod
    def __prepare_angles_contents(angles: Optional[dict],
                                  elements: list) -> list:
        """
        Prepare the angles parameters contnet of the force field file

        Parameters
        ----------
        angles
            angles parameters

        elements
            elements in the force field

        Returns
        -------
        angles_contents
            angles parameters content of the force field file

        See Also
        --------
        ReactiveForceFieldWriter.__get_num_from_str
        """

        angles_contents = []

        number_of_angles = len(angles) if angles is not None else 0

        angles_contents.append(
            ' {:^2}'.format(number_of_angles) +
                    ' ! Nr of angles;' +
                    'at1;at2;at3;Thetao,o;ka;kb;pv1;pv2;val(bo)\n')

        if number_of_angles:

            for key, values in angles.items():

                num = ReactiveForceFieldWriter.__get_num_from_str(elements,
                                                                  key)

                angles_contents.append(
                    '  ' + num + ' ' * 2 +
                            str(values['value']).lstrip('[').rstrip(']') +
                            '\n')

        return angles_contents

    @staticmethod
    def __prepare_torsions_contents(torsions: Optional[dict],
                                    elements: list) -> list:
        """
        Prepare the torsions parameters contnet of the force field file

        Parameters
        ----------
        torsions
            torsions parameters

        elements
            elements in the force field

        Returns
        -------
        torsions_contents
            torsions parameters content of the force field file

        See Also
        --------
        ReactiveForceFieldWriter.__get_num_from_str
        """

        torsions_contents = []

        number_of_torsions = len(torsions) if torsions is not None else 0

        torsions_contents.append(
            ' {:^2}'.format(number_of_torsions) +
                    ' ! Nr of torsions;' +
                    'at1;at2;at3;at4;;V1;V2;V3;V2(BO);vconj;n.u;n\n')

        if number_of_torsions:

            for key, values in torsions.items():

                num = ReactiveForceFieldWriter.__get_num_from_str(elements,
                                                                  key)

                torsions_contents.append(
                    '  ' + num + ' ' * 2 +
                    str(values['value']).lstrip('[').rstrip(']') +
                    '\n')

        return torsions_contents

    @staticmethod
    def __prepare_hydrogen_bonds_contents(hydrogen_bonds: Optional[dict],
                                          elements: list) -> list:
        """
        Prepare the hydrogen bonds parameters contnet of the force field file

        Parameters
        ----------
        hydrogen_bonds
            hydrogen bonds parameters

        elements
            elements in the force field

        Returns
        -------
        hydrogen_bonds_contents
            hydrogen bonds parameters content of the force field file
        """

        hydrogen_bonds_contents = []

        number_of_hydrogen_bonds = (
            len(hydrogen_bonds) if hydrogen_bonds is not None else 0)

        hydrogen_bonds_contents.append(
            ' {:^2}'.format(number_of_hydrogen_bonds) +
            ' ! Nr of hydrogen bonds;at1;at2;at3;Rhb;Dehb;vhb1')

        if number_of_hydrogen_bonds:

            hydrogen_bonds_contents.append('\n')

            for key, values in hydrogen_bonds.items():

                num = ReactiveForceFieldWriter.__get_num_from_str(
                    elements, key)

                hydrogen_bonds_contents.append(
                    f"  {num}  "
                    f"{str(values['value']).lstrip('[').rstrip(']')}\n")

        return hydrogen_bonds_contents

    @staticmethod
    def __get_num_from_str(elements: list, string: str) -> str:
        """
        Convert the bonds string from elements to indices

        e.g.S-Ge-H where S = 1, Ge = 2, H =3 to 1 2 3

        Parameters
        ----------
        elements
            elements in the force field

        string
           the bond string

        Returns
        -------
        bond string in indices

        """

        num = list()

        element_list = list(elements)

        for atom in string.split('-'):

            if atom == '*':
                num.append('0')
            else:
                num.append(f'{element_list.index(atom) + 1}')

        return '  '.join(num)


def save_reactive_force_field(parameters: dict, directory: str) -> None:
    """
    Save the force field parameters to the file

    Parameters
    ----------
    parameters
        force field parameters

    directory
        target directory

    Returns
    -------
    None

    See Also
    --------
    cores.utilities.file_path.file_create_directory
    ReactiveForceFieldWriter
    """

    file_create_directory(os.path.dirname(directory))

    writer = ReactiveForceFieldWriter().write_force_field_to_file(
        parameters, directory)
