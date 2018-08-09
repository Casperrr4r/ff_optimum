# -*- coding: utf-8 -*-
import copy
import os
import platform

from lammps import lammps
import numpy as np

from ff_optimum.user_packages.reaxff.io import save_parameters_to_file
from ff_optimum.cores.utilities import (
    EventLogger, file_set_path, get_client, is_client_ready)


logger = EventLogger(__name__)

__all__ = ['compute_values_lammps']


REAXFF_SUBENERGY = {'eb': 1, 'ea': 2, 'elp': 3, 'ev': 5,
                    'epen': 6, 'ecoa': 7, 'ehb': 8, 'et': 9,
                    'eco': 10, 'ew': 11, 'ep': 12, 'eqeq': 14}


def __extract_sub_energy_from_lammps(lmp: lammps) -> dict:
    """
    Get the reaxff sub energy from LAMMPS

    Parameters
    ----------
    lmp
        reference of LAMMPS object

    Returns
    -------
    sub_energy
        dictionary containing the sub energy

    """

    sub_energy = dict.fromkeys(REAXFF_SUBENERGY.keys(), 0)

    for key in sub_energy.keys():
        sub_energy[key] = lmp.extract_variable(key, "1", 0)

    return sub_energy


def compute_values_lammps(lammps_commands_dict: dict,
                          parameters: dict, temp_directory: str) -> dict:
    """
    Compute values of multi-frame of input lammps_commands by LAMMPS

    If the Singleton Client of ipyparallel is ready the value will be
    calulated parallely otherwise, it will be calculated serially.

    Parameters
    ----------
    lammps_commands_dict
          dictionary of list containing the LAMMPS commands according to the
          molecule name

    parameters
          ReaxFF parameters

    Returns
    -------
    calculated_values
          dictionary of list containing the calculated value according to the
          molecule name

    See Also
    --------
    reaxff.compute.compute_values.__compute_values_lammps_multi_frame_parallel
    reaxff.compute.compute_values.__compute_values_lammps_multi_frame_serial
    reaxff.io.write_force_field.save_force_field_to_file
    utilities.parallel.ipyparallel_singleton.is_client_ready
    """

    force_field_path = file_set_path(temp_directory, 'ffield_temp')

    save_parameters_to_file(parameters, force_field_path)

    if is_client_ready():
        return __compute_values_lammps_multi_frame_parallel(
            lammps_commands_dict)

    return __compute_values_lammps_multi_frame_serial(lammps_commands_dict)


def __compute_values_lammps_multi_frame_serial(
        lammps_commands_dict: dict) -> dict:
    """
    Compute values of multi-frame of input lammps_commands by LAMMPS in serial

    Parameters
    ----------
    lammps_commands_dict
          dictionary of list containing the LAMMPS commands according to the
          molecule name

    Returns
    -------
    calculated_values
          dictionary of list containing the calculated value according to the
          molecule name

    See Also
    --------
    reaxff.compute.compute_values.__compute_values_lammps_one_frame
    """

    calculated_values = {}

    for molecule_name in lammps_commands_dict.keys():
        calculated_values[molecule_name.lower()] = \
            list(map(__compute_values_lammps_one_frame,
                     lammps_commands_dict[molecule_name]))

    return calculated_values


def __compute_values_lammps_multi_frame_parallel(
        lammps_commands_dict: dict) -> dict:
    """
    Compute values of multi-frame of input lammps_commands by LAMMPS parallely
    by ipyparallel

    Parameters
    ----------
    lammps_commands_dict
          dictionary of list containing the LAMMPS commands according to the
          molecule name

    Returns
    -------
    calculated_values
          dictionary of list containing the calculated value according to the
          molecule name

    See Also
    --------
    reaxff.compute.compute_values.__compute_values_lammps_one_frame
    utilities.parallel.ipyparallel_singleton.get_client
    """

    lmp_cmd_list = []

    calculated_values = {}

    direct_view = get_client()[:]

    for cmds in lammps_commands_dict.values():
        lmp_cmd_list.extend(cmds)

    for molecule_name in lammps_commands_dict.keys():
        calculated_values[molecule_name] = []

    sync_res = direct_view.map_sync(__compute_values_lammps_one_frame,
                                    lmp_cmd_list)

    results = copy.deepcopy(sync_res)

    for res in results:
        calculated_values[res['name'].lower()].append(res)

    return calculated_values


def __compute_values_lammps_one_frame(lammps_commands: list) -> dict:
    """
    Compute values of one frame of input lammps_commands by LAMMPS

    Parameters
    ----------
    lammps_commands
          list containing one frame of command

    Returns
    -------
    results
          dictionary containing name, step, charge, forces, stress and
          energy calculated by LAMMPS

    See Also
    --------
    reaxff.compute.compute_values.__retrive_trace_info
    reaxff.compute.compute_values.__extract_charge_from_lammps
    reaxff.compute.compute_values.__extract_forces_from_lammps
    reaxff.compute.compute_values.__extract_stress_from_lammps
    reaxff.compute.compute_values.__set_flag_from_trace_info
    """

    _, flags, name_and_step = __retrive_trace_info(lammps_commands)

    try:

        res = None

        lmp = lammps("", ["-screen", "none", "-log", "none", "-nocite"])

        for lammps_command in lammps_commands.values():
            for cmd in lammps_command:
                lmp.command(cmd)

        lmp.command("variable etot equal etotal")

        natoms = lmp.get_natoms()

        forces = (__extract_forces_from_lammps(lmp, natoms)
                  if flags[0] else None)

        charge = (__extract_charge_from_lammps(lmp, natoms)
                  if flags[1] else None)

        stress = __extract_stress_from_lammps(lmp) if flags[2] else None

        total_energy = (lmp.extract_fix("2", 0, 0) if flags[3]
                        else lmp.extract_variable("etot", "1", 0) / natoms)

    except Exception as e:

        logger.error(e)

        logger.error(f'{name_and_step[0]} calculation failed')

    else:

        res = {'name': name_and_step[0], 'step': name_and_step[1],
               'q': charge, 'fx': forces[0], 'fy': forces[1], 'fz': forces[2],
               'stress': stress, 'energy': total_energy}

    finally:

        if lmp is not None:
            lmp.close()

        return res


def __extract_charge_from_lammps(lmp: lammps,
                                 number_of_atoms: int) -> np.ndarray:
    """
    Extract charge calculated by LAMMPS

    Parameters
    ----------
    lmp
          reference of the lammps object

    number_of_atoms
          number of atoms stated in the trace info of the input command

    Returns
    -------
    charge
          charge calculated by LAMMPS

    """

    extracted_q = lmp.extract_atom("q", 2)

    q = np.ctypeslib.as_array(extracted_q, (1, number_of_atoms))

    charge = q[0].copy()

    return charge


def __extract_forces_from_lammps(lmp: lammps, number_of_atoms: int,
                                 no_of_dimension: int=3) -> np.ndarray:
    """
    Extract forces calculated by LAMMPS

    Parameters
    ----------
    lmp
          reference of the lammps object

    number_of_atoms
          number of atoms stated in the trace info of the input command

    Returns
    -------
    forces
          forces calculated by LAMMPS

    """

    f_extracted = lmp.extract_atom("f", 3)

    forces = np.zeros((number_of_atoms, no_of_dimension))

    indices = np.arange(no_of_dimension)

    for index in range(number_of_atoms):
        np.put(forces, indices + no_of_dimension * index,
               np.ctypeslib.as_array(f_extracted[index], (no_of_dimension,)))

    return forces.T


def __extract_stress_from_lammps(lmp: lammps) -> np.ndarray:
    """
    Extract stress calculated by LAMMPS

    Parameters
    ----------
    lmp
          reference of the lammps object

    Returns
    -------
    stress
          stress calculated by LAMMPS

    """

    stress = np.zeros(9)

    np.put(stress, 0, -1 * lmp.extract_variable("pxx", "1", 0))

    np.put(stress, 1, -1 * lmp.extract_variable("pxy", "1", 0))

    np.put(stress, 2, -1 * lmp.extract_variable("pxz", "1", 0))

    np.put(stress, 3, np.take(stress, 1))

    np.put(stress, 4, -1 * lmp.extract_variable("pyy", "1", 0))

    np.put(stress, 5, -1 * lmp.extract_variable("pyz", "1", 0))

    np.put(stress, 6, np.take(stress, 2))

    np.put(stress, 7, np.take(stress, 5))

    np.put(stress, 8, -1 * lmp.extract_variable("pzz", "1", 0))

    return stress


def __retrive_trace_info(lammps_commands: dict) -> tuple:
    """
    Retrive the trace info from compiled commands

    Parameters
    ----------
    lammps_commands
        compiled LAMMPS commands

    Returns
    -------
    natoms_in_cmd
        number of atoms stated in the command

    flags
        Force, Charge, Stress, Temperature
        True if the value will be retrived

    name_and_step
        molecule name and step number

    """

    trace_info = list(map((lambda x: x.split(': ')[1]),
                          lammps_commands['trace_info'][:-1]))

    flags = ['True' in info for info in trace_info[1:5]]

    name_and_step = (
        lammps_commands['trace_info'][-1].split('#')[-1].split(' '))

    return (int(trace_info[0]), flags, name_and_step)
