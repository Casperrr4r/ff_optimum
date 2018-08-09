# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from ff_optimum.cores.compute_builtin import compute_error_builtin, Fitness
from ff_optimum.cores.utilities.parallel_singleton import (
    get_client, is_client_ready)
from ff_optimum.user_packages.reaxff.optimization_setting.objectives import (
    ReaxFFObjectives)
from ff_optimum.user_packages.reaxff.visualize import (
    print_reaxff_error_to_screen, print_formation_energy_header,
    print_formation_energy_one_mol)


__all__ = ['compute_error_reaxff']


def compute_error_reaxff(calculated_values: dict,
                         training_datas: dict,
                         settings: dict,
                         output_directory: str=None) -> Union[float, Fitness]:
    """
    Compute the error between LAMMPS calculated values and DFT training datas

    Parameters
    ----------
    calculated_values
        values calculated by LAMMPS

    training_datas
        DFT training data

    settings
        optimization settings

    Returns
    -------
    weighted_sum_error
        if it is scalarized objective function

    Fitness object
        Otherwise

    See Also
    --------
    __compute_error_reaxff_inner

    """

    weights = settings['weights']

    fitness, total_errors = __compute_error_reaxff_inner(
        calculated_values, training_datas, settings)

    if weights is not None:

        weighted_sum_error = 0

        for error, weight in zip(total_errors, weights):
            weighted_sum_error += error * weight

        return weighted_sum_error

    return Fitness(fitness.keys(), np.fromiter(fitness.values(), dtype='f8'))


def __compute_error_reaxff_one_objective(
        calculated_values: dict, training_datas: dict,
        settings: dict, objective_name: str,
        objective_function: str) -> Union[dict, np.ndarray]:
    """
    Compute the error between LAMMPS calculated values
    and DFT training datas of one objective

    Parameters
    ----------
    calculated_values
        values calculated by LAMMPS

    training_datas
        DFT training data

    settings
        optimization settings

    objective_name
        name of the objective

    objective_function
        objective function of the objective

    Returns
    -------
    objective_value
        if the objecive value is not nan

    maximumu of 64 bit floating point number
        otherwise

    See Also
    --------
    __compute_error_in_formation_energy

    """

    objective_value = np.nan

    for function_name in REAXFF_OBJECTIVE_FUNCTIONS.keys():
        if objective_name.endswith(function_name.lower()):

            mol_name = objective_name.split(f'_{function_name}')[0].lower()

            if objective_name.endswith('formation'):
                objective_value = (
                    __compute_error_in_formation_energy(
                        calculated_values, training_datas,
                        mol_name, objective_function,
                        settings))
            else:
                objective_value = (
                    REAXFF_OBJECTIVE_FUNCTIONS[function_name](
                        calculated_values[mol_name],
                        training_datas[mol_name],
                        objective_function))

    return (np.finfo('f8').max if np.isnan(objective_value)
            else objective_value)


def __compute_total_reaxff_error(fitness: dict) -> np.ndarray:
    """
    Compute total error from fitness

    Parameters
    ----------
    fitness
        dictionary object contains all the objective function values

    Returns
    -------
    average of total_errors

    """

    total_errors = np.zeros(len(ReaxFFObjectives))

    counts = np.zeros(len(ReaxFFObjectives))

    for fitness_name, fitness_value in fitness.items():
        for function_name in REAXFF_OBJECTIVE_FUNCTIONS.keys():
            if fitness_name.endswith(function_name.lower()):

                idx = ReaxFFObjectives[function_name.lower()].value

                if fitness_value:
                    total_errors[idx] += fitness_value

                    counts[idx] += 1

    return total_errors / np.where(counts > 0, counts, 1)


def __compute_error_reaxff_inner(calculated_values: dict,
                                 training_datas: dict,
                                 settings: dict) -> Union[dict, np.ndarray]:
    """
    Helper function for computing reaxff error ipyparallel will be used
    if the client is ready

    Parameters
    ----------
    calculated_values
        values computed by LAMMPS

    training_datas
        DFT training data

    settings
        optimization settings

    Returns
    -------
    fitness

    total_errors

    See Also
    --------
    __compute_error_reaxff_one_objective
    __compute_total_reaxff_error
    user_packages.reaxff.visualize.print_formation_energy_header
    user_packages.reaxff.visualize.print_formation_energy_one_mol
    user_packages.reaxff.visualize.print_reaxff_error_to_screen
    """

    fitness = {}

    objectives, slient = settings['objectives'], settings['slient']

    if is_client_ready() and slient:

        direct_view = get_client()[:]

        results = direct_view.map_sync(
            lambda objective: (
                objective[0], __compute_error_reaxff_one_objective(
                    calculated_values, training_datas, settings,
                    objective[0], objective[1])),
            objectives.items())

        for res in results:
            fitness[res[0]] = res[1]
    else:

        if not slient:
            print_formation_energy_header()

        for objective_name, objective_function in objectives.items():
            fitness[objective_name] = __compute_error_reaxff_one_objective(
                calculated_values, training_datas, settings,
                objective_name, objective_function)

    total_errors = __compute_total_reaxff_error(fitness)

    if not slient:
        print_formation_energy_one_mol(
            'TOTAL AVG', '', '', '',
            total_errors[ReaxFFObjectives['formation'].value])

        print_reaxff_error_to_screen(fitness, settings, total_errors)

    return fitness, total_errors


def __compute_error_in_energy(calculated_values: list,
                              training_datas: list,
                              error_type: str) -> float:
    """
    Compute error in energy

    calculated_values
        values calculated by LAMMPS

    training_datas
        DFT training data

    error_type
        type of error function
    """

    target_energies = np.zeros(len(calculated_values))

    calculated_eneregies = np.zeros(len(calculated_values))

    for idx in range(len(calculated_values)):

        np.put(target_energies, idx, training_datas[idx]['energy'])

        np.put(calculated_eneregies, idx, calculated_values[idx]['energy'])

    min_idx = np.argmin(target_energies)

    target_energies -= np.take(target_energies, min_idx)

    calculated_eneregies -= np.take(calculated_eneregies, min_idx)

    return compute_error_builtin(calculated_eneregies,
                                 target_energies, error_type)


def __compute_formation_energy(energies: dict, mol_name: str,
                               dependencies: list, function: str) -> float:
    """
    Compute the formation energy

    Parameters
    ----------

    energies
        calculated energies either by LAMMPS or DFT


    mol_name
        name of the molecule to be computed

    dependencies
        molecules needed to compute the formation energy

    function
        expression of the formation energy

    Returns
    -------
    formation energy

    See Also
    --------
    __get_minimum_energy
    """

    min_energies = {}

    min_energies[mol_name] = (
        __get_minimum_energy(energies[mol_name]))

    for dependency in dependencies[mol_name]:
        min_energies[dependency] = (
            __get_minimum_energy(energies[dependency]))

    return eval(function, None, {'energy': min_energies})


def __compute_error_in_formation_energy(
        calculated_values: dict, training_datas: dict, mol_name: str,
        function: str, setting: dict) -> float:
    """
    Compute the eror in formation energy

    Parameters
    ----------
    calculated_values
        values computed by LAMMPS

    training_datas
        DFT traininng datas

    mol_name
        Name of the molecule

    function
        the function to be used for computing formation energy

    setting
        optimization setting

    See Also
    --------
    __get_minimum_energy
    __compute_formation_energy
    """

    dependencies = setting['dependencies']

    mol_weight = setting['mol_weights'].get(mol_name, 1.0)

    if dependencies[mol_name] is not None:

        calculated_formation = __compute_formation_energy(
            calculated_values, mol_name, dependencies, function)

        target_formation = __compute_formation_energy(
            training_datas, mol_name, dependencies, function)

    else:

        calculated_formation = __get_minimum_energy(
            calculated_values[mol_name])

        target_formation = eval(function)

    error = np.abs(calculated_formation - target_formation)

    if not setting['slient']:
        print_formation_energy_one_mol(
            mol_name, setting['composition'], calculated_formation,
            target_formation, error)

    return error * mol_weight


def __get_minimum_energy(molecule: list) -> float:
    """
    Get the minimum energy from the list of steps

    Parameters
    ----------
    molecule
        the values of a particular molecule

    Returns
    -------
    Minimum energy
    """

    return min(molecule[frame]['energy'] for frame in range(len(molecule)))


def __compute_error_from_lammps_results(calculated_values: list,
                                        training_datas: list,
                                        key: str,
                                        error_type: str) -> float:
    """
    Compute error of an objective

    Parameters
    ----------
    calculated_values
        calculated values of the molecule

    training_datas
        training data of the molecule

    key
        objective type

    error_type
        type of the error function

    Returns
    -------
    the computed error
    """

    calculated_value, target_value = [], []

    for idx in range(len(training_datas)):

        if calculated_values[idx].get(key) is None:
            return None

        calculated_value.append(calculated_values[idx].get(key))

        target_value.append(training_datas[idx].get(key))

    return compute_error_builtin(np.array(calculated_value),
                                 np.array(target_value), error_type)


def __compute_error_in_charge(calculated_values: list,
                              training_datas: list,
                              error_type) -> float:
    """
    Compute error in charge

    Parameters
    ----------
    calculated_values
        calculated values of the molecule

    training_datas
        training data of the molecule

    error_type
        type of the error function

    Returns
    -------
    error in charge

    See Also
    --------
    __compute_error_from_lammps_results
    """

    return __compute_error_from_lammps_results(calculated_values,
                                               training_datas, 'q', error_type)


def __compute_error_in_force(calculated_values: list,
                             training_datas: list,
                             error_type: str) -> float:
    """
    Compute error in stress

    Parameters
    ----------
    calculated_values
        calculated values of the molecule

    training_datas
        training data of the molecule

    error_type
        type of the error function

    Returns
    -------
    error in force

    See Also
    --------
    __compute_error_from_lammps_results
    """

    error_fx = __compute_error_from_lammps_results(
        calculated_values, training_datas, 'fx', error_type)

    error_fy = __compute_error_from_lammps_results(
        calculated_values, training_datas, 'fy', error_type)

    error_fz = __compute_error_from_lammps_results(
        calculated_values, training_datas, 'fz', error_type)

    return (error_fx + error_fy + error_fz) / 3


def __compute_error_in_stress(calculated_values: list,
                              training_datas: list,
                              error_type) -> float:
    """
    Compute error in stress

    Parameters
    ----------
    calculated_values
        calculated values of the molecule

    training_datas
        training data of the molecule

    error_type
        type of the error function

    Returns
    -------
    error in stress

    See Also
    --------
    __compute_error_from_lammps_results
    """

    return __compute_error_from_lammps_results(calculated_values,
                                               training_datas, 'stress',
                                               error_type)


REAXFF_OBJECTIVE_FUNCTIONS = {'charge': __compute_error_in_charge,
                              'energy': __compute_error_in_energy,
                              'formation': __compute_error_in_formation_energy,
                              'force': __compute_error_in_force,
                              'stress': __compute_error_in_stress}
