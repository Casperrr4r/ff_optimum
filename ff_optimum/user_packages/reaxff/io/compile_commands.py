# -*- coding: utf-8 -*-
from typing import List, Union

import numpy as np

from .atomic_mass import get_element_name_from_atomic_mass
from ff_optimum.user_packages.reaxff.compute.simulation_box import (
    SimulationBox)
from ff_optimum.cores.utilities import argument_type_check

__all__ = ['compile_lammps_commands']


@argument_type_check
def compile_lammps_commands(
        xml_result: dict, flags: dict, reaxff_path: str) -> list:
    """
    Compile xml parse results to LAMMPS commands

    Parameters
    ----------
    xml_result
        xml parse results of the DFT xml training data

    flags
        flags of the training data

    reaxff_path
        path of the ReaxFF parameters file


    See Also
    --------
    LammpsCommandCompiler
    """

    with LammpsCommandCompiler() as compiler:

        compiler.compile_command_from_xml_result(
            xml_result, flags, reaxff_path)

        return compiler.get_compiled_command()


class LammpsCommandCompiler(object):

    """
    Class for compiling DFT training data xml results to LAMMPS commands

    Attributes
    ----------
    __trace_infos: list
    __units_command: list
    __atom_style_command: list
    __boundary_command: list
    __region_box_commands: list
    __create_box_command: list
    __create_atom_commands: list
    __mass_command: list
    __replicate_commands: list
    __pair_style_command: list
    __pair_coeff_command: list
    __variable_commands: list
    __fix_and_run_command: list

    Methods
    -------
    compile_command_from_xml_result(xml_result, flag, force_field_path)
        compile the DFT training data xml results to LAMMPS commands

    get_compiled_command()
        return the compiled commands
    """

    __slots__ = ['__trace_infos', '__units_command', '__atom_style_command',
                 '__boundary_command', '__region_box_commands',
                 '__create_box_command', '__create_atom_commands',
                 '__mass_command', '__replicate_commands',
                 '__pair_style_command', '__pair_coeff_command',
                 '__variable_commands', '__fix_and_run_command']

    def __init__(self):

        self.__trace_infos = None

        self.__units_command = None

        self.__atom_style_command = 'atom_style full'

        self.__boundary_command = None

        self.__region_box_commands = None

        self.__create_box_command = None

        self.__create_atom_commands = None

        self.__mass_command = None

        self.__replicate_commands = None

        self.__pair_style_command = 'pair_style reax/c NULL checkqeq yes'

        self.__pair_coeff_command = None

        self.__variable_commands = None

        self.__fix_and_run_command = (
            ['fix 1 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c', 'run 0'])

    def __enter__(self) -> object:
        return self

    def __exit__(self, exc_ty, exc_val, tb) -> None:
        pass

    @argument_type_check
    def compile_command_from_xml_result(
            self, xml_result: dict, flag: dict, force_field_path: str) -> None:
        """
        Compile xml parse results to LAMMPS commands

        Parameters
        ----------
        xml_result
            xml parse results of the DFT xml training data

        flags
            flags of the training data

        force_field_path
            path of the ReaxFF parameters file

        See Also
        --------
        LammpsCommandCompiler.__get_trace_info_from_step
        LammpsCommandCompiler.__get_units_command_from_result
        LammpsCommandCompiler.__get_boundary_command_from_result
        LammpsCommandCompiler.__get_create_box_command_from_result
        LammpsCommandCompiler.__get_create_atom_commands_from_step
        LammpsCommandCompiler.__get_mass_commands_from_result
        LammpsCommandCompiler.__get_replicate_command_from_step
        LammpsCommandCompiler.__get_pair_coeff_command_from_result
        LammpsCommandCompiler.__get_variable_commands_from_result
        """

        self.__trace_infos = \
            [self.__get_trace_info_from_step(xml_result, flag, step_no)
                for step_no in range(len(xml_result['positions']))]

        self.__units_command = \
            self.__get_units_command_from_result(xml_result)

        self.__boundary_command = \
            self.__get_boundary_command_from_result(xml_result)

        self.__region_box_commands = \
            [self.__get_region_box_command_from_step(step)
                for step in xml_result['positions']]

        self.__create_box_command = \
            self.__get_create_box_command_from_result(xml_result)

        self.__create_atom_commands = \
            [self.__get_create_atom_commands_from_step(step)
                for step in xml_result['positions']]

        self.__mass_command = \
            self.__get_mass_commands_from_result(xml_result)

        self.__replicate_commands = \
            [self.__get_replicate_command_from_step(step)
                for step in xml_result['positions']]

        self.__pair_coeff_command = \
            self.__get_pair_coeff_command_from_result(
                xml_result, force_field_path)

        self.__variable_commands = \
            self.__get_variable_commands_from_result(xml_result)

    def get_compiled_command(self) -> List:
        """
        Get the compiled command

        Returns
        -------
        List of compiled commands
        """

        return list(
            map(lambda x:
                self.__get_compiled_command_one_step(
                x[0], x[1], x[2], x[3]),
                zip(self.__trace_infos, self.__region_box_commands,
                    self.__create_atom_commands,
                    self.__replicate_commands)))

    def __get_compiled_command_one_step(
        self, trace_info: List[str],
                region_box_command: str,
                create_atom_command: str,
                replicate_command: str) -> dict:
        """
        Get one step of compiled command

        Parameters
        ----------
        region_box_command
            command of the region box

        create_atom_command
            commands of create atom

        replicate_command
            commands of replicate

        Returns
        -------
        dictionary of one step of compiled command
        """

        commands_before_pair_coeff = list()

        commands_after_pair_coeff = list()

        commands_before_pair_coeff.append(self.__units_command)

        commands_before_pair_coeff.append(self.__atom_style_command)

        commands_before_pair_coeff.extend(self.__boundary_command)

        commands_before_pair_coeff.append(region_box_command)

        commands_before_pair_coeff.append(self.__create_box_command)

        commands_before_pair_coeff.extend(create_atom_command)

        commands_before_pair_coeff.extend(self.__mass_command)

        commands_before_pair_coeff.append(self.__pair_style_command)

        commands_after_pair_coeff.extend(self.__variable_commands)

        commands_after_pair_coeff.extend(self.__fix_and_run_command)

        lammps_commands = {'trace_info': trace_info,
                           'before_pair_coeff': commands_before_pair_coeff,
                           'pair_coeff': [self.__pair_coeff_command],
                           'after_pair_coeff': commands_after_pair_coeff}

        return lammps_commands

    @staticmethod
    def __get_trace_info_from_step(
            xml_result: dict, flag: dict, no_of_step: int) -> List[str]:
        """
        Get trace information from xml parse results

        Parameters
        ----------
        xml_result
            xml parse results

        flag
            flag of the training data

        no_of_step
         the current step number
        """
        step = xml_result['positions'][no_of_step]

        command = []

        command.append(f'#number_of_atom: {len(step["type"])}')

        command.append(f'#f_flag: {flag["fx"]}')

        command.append(f'#q_flag: {flag["q"]}')

        command.append(f'#stress_flag: {flag["stress"]}')

        command.append(f'#t_flag: False')

        command.append(f'#{xml_result["name"]} {no_of_step+1}')

        return command

    @staticmethod
    def __get_units_command_from_result(xml_result: dict) -> str:
        """
        Get units command from xml parse results

        Parameters
        ----------
        xml_result
            xml parse results

        Returns
        -------
        unit command
        """

        return f'units {xml_result["unit"]}'

    @staticmethod
    def __get_boundary_command_from_result(
            xml_results: dict) -> Union[List[str], str]:
        """
        Get boundary command from xml parse results

        Parameters
        ----------
        xml_result
            xml parse results

        Returns
        -------
        boundary command

        """

        if xml_results['positions'][0]['box'] is not None:
            return [f'boundary {"".join(xml_results["periodicity"])}',
                    'box tilt large']

        return ['boundary f f f']

    @staticmethod
    def __get_region_box_command_from_step(step: dict) -> str:
        """
        Get region box command from xml parse results

        Parameters
        ----------
        step
            the step information in xml parse results

        Returns
        -------
        box_command

        """

        if step['box'] is None:
            return 'region box prism {"-10 10 " * 3 }{"0.0 " * 3}'

        box_command = 'region box prism'

        box = SimulationBox(step['box'])

        xyz_box = box.get_xyz_box()

        for idx in range(xyz_box.size):
            box_command += ' 0 ' + str(xyz_box[idx]) if idx < 3 else ' ' + \
                str(xyz_box[idx])
        return box_command

    @staticmethod
    def __get_create_box_command_from_result(xml_result: dict) -> str:
        """
        Get create box command from xml parse results

        Parameters
        ----------
        xml_result
            xml parse results of training data

        Returns
        -------
        create box command

        """
        return f'create_box {len(xml_result["types"])} box'

    @staticmethod
    def __get_create_atom_commands_from_step(step: dict) -> list:
        """
        Get create atom commands from xml parse results

        Parameters
        ----------
        step
            the step information in xml parse results

        Returns
        -------
        create atoms commands

        """

        box = SimulationBox(step['box'])

        xyz_box = box.get_xyz_box()

        lmp_coordinates = map(box.convert_xml_coordinate_to_lmp_coordinate,
                              zip(step['x'][0], step['y'][0], step['z'][0]))

        create_atoms_commands = list(map(lambda x: (f'create_atoms {x[0]} '
                                                    f'single {x[1][0]} '
                                                    f'{x[1][1]} {x[1][2]}'),
                                         zip(step['type'], lmp_coordinates)))

        return create_atoms_commands

    @staticmethod
    def __get_mass_commands_from_result(xml_result: dict) -> List[str]:
        """
        Get mass command from xml parse results

        Parameters
        ----------
        xml_result
            xml parse results of training data

        Returns
        -------
        mass command

        """
        return list(map(lambda x: f'mass {x[0]} {x[1]}',
                        zip(xml_result['types'],
                            xml_result['mass'].split(' '))))

    def __get_replicate_command_from_step(self, step: dict) -> str:
        """
        Get replicate command from xml parse results

        Parameters
        ----------
        step
            the step information in xml parse results

        Returns
        -------
        replicate command

        See Also
        --------
        LammpsCommandCompiler.__replicate

        """

        if (step['box'] is not None and
                not self.__boundary_command[0].endswith('f f f')):

            xyz_box = SimulationBox(step['box']).get_xyz_box()

            if np.less(xyz_box[0:3], 3.0).any():

                rep = np.vectorize(self.__replicate)(xyz_box[0:3])

                periodicities = self.__boundary_command[0].split(' ')[-3:]

                for idx in range(3):
                    if 'f' in periodicities[idx]:
                        rep[idx] = 1

                return f'replicate {rep[0]} {rep[1]} {rep[2]}'

        return str()

    @staticmethod
    def __get_pair_coeff_command_from_result(
            xml_result: dict, force_field_path: str) -> str:
        """
        Get the pair coefficient commands from the xml parse result

        Parameters
        ----------
        xml_result
            dictionary object contain the xml parse result

        force_field_path
            file path of the force field files

        """

        elements = map(get_element_name_from_atomic_mass,
                       xml_result['mass'].split(' '))

        return f'pair_coeff * * {force_field_path} {" ".join(elements)}'

    @staticmethod
    def __get_variable_commands_from_result(xml_result: dict) -> List[str]:
        """
        Get the variable commands from the xml parse result

        Parameters
        ----------
        xml_result
            dictionary object contain the xml parse result

        Returns
        -------
        variable_commands
        """

        variable_commands = ['compute reax all pair reax/c']

        stress = ['pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz']

        REAXFF_SUBENERGY = {'eb': 1, 'ea': 2, 'elp': 3, 'ev': 5,
                            'epen': 6, 'ecoa': 7, 'ehb': 8, 'et': 9,
                            'eco': 10, 'ew': 11, 'ep': 12, 'eqeq': 14}

        if xml_result['positions'][0]['box'] is not None:

            for subenergy, index in REAXFF_SUBENERGY.items():
                variable_commands.append(f'variable {subenergy} '
                                         f'equal c_reax[{index}]')

            variable_commands.extend(
                map(lambda x: f'variable {x} equal {x}', stress))

            variable_commands.append(
                'thermo_style custom step etotal '
                f'v_{" v_".join(REAXFF_SUBENERGY.keys())} '
                f'{" ".join(stress)}')

        return variable_commands

    @staticmethod
    def __replicate(dim: float) -> float:
        """
        Replicate

        Parameters
        ----------
        dim

        Returns
        -------
        rep
        """

        rep, dim0 = 1, dim

        while dim / 2.0 < 3.0:

            rep += 1

            dim = rep * dim0

        return rep
