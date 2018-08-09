# -*- coding: utf-8 -*-
from ff_optimum.user_packages.reaxff.optimization_setting.objectives import (
    ReaxFFObjectives)

__all__ = ['print_reaxff_error_to_screen',
           'print_formation_energy_header',
           'print_formation_energy_one_mol']

ENERGY_UNIT = 'kcal/mol'


def __print_dash(length: int=72) -> None:
    """
    Print '-' to the screen

    Parameters
    ----------
    length
        number of '-'

    Returns
    -------
    None
    """

    print('-' * length)


def print_reaxff_error_to_screen(fitness: dict, settings,
                                 total_errors: list) -> None:
    """
    Print the error between DFT training data and ReaxFF values to the screen

    Parameters
    ----------
    fitness


    settings

    total_errors

    See Also
    --------
    __output_reaxff_error_header
    __output_reaxff_error_one_mol

    """

    fitness_mapping_by_mol_name = dict()

    for fitness_name in fitness.keys():

        for objective in ReaxFFObjectives:

            if fitness_name.endswith(objective.name):
                mol_name = fitness_name.split(f'_{objective.name}')[0]

                if fitness_mapping_by_mol_name.get(mol_name, None) is None:
                    fitness_mapping_by_mol_name[mol_name] = [None] * 5

                idx = objective.value

                fitness_mapping_by_mol_name[mol_name][idx] = (
                    fitness[fitness_name])

    __output_reaxff_error_header('err')

    for mol_name, fitness in fitness_mapping_by_mol_name.items():
        __output_reaxff_error_one_mol('err', mol_name, fitness)
    else:
        __output_reaxff_error_one_mol('err', "TOTAL AVG", total_errors)


def print_formation_energy_header() -> None:
    """
    Print the header of formation energy

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    header = (f'COMB   # {"Combination":50} {" "*5}Trial |{" "*5}Target'
              f'    {ENERGY_UNIT} (Errors)')

    print(header)

    __print_dash(104)


def print_formation_energy_one_mol(
        molecule_name: str, composition: str,
        trial: float, target: float, error: float) -> None:
    """
    Print the formation energy of one molecule

    Parameters
    ----------
    molecule_name
        Name of the molecule

    composition
        Composition of calculating the formation energy

    trial
        Formation energy computed by ReaxFF values

    target
        Formation energy computed by DFT training data

    error
        the difference between the trial and target

    """

    if "TOTAL AVG" in molecule_name:

        __print_dash(104)

        one_mol = (f'COMB {" " * 3} {molecule_name:50} {" " * 10} | '
                   f'{" " * 10}    {ENERGY_UNIT} ({error:5.4f})')

    else:
        formula = f'{molecule_name} -> {composition[molecule_name]}'

        one_mol = (f'COMB {" " * 3} {formula:50} {trial:10.4f} | '
                   f'{target:10.4f}    {ENERGY_UNIT} ({error:5.4f})')

    print(one_mol)


def __output_reaxff_error_header(error_type: str) -> None:

    header = (f'{error_type.upper()} Name{" " * 20} Energy'
              f'{" " * 4} Charge {" " * 4} Force {" " * 2}Pressure')

    print(header)

    __print_dash()


def __output_reaxff_error_one_mol(error_type: str, molecule_name: str,
                                  errors: list) -> None:
    """
    Output errors of one molecule to screen including charges, total energies,
    forces and stresses

    Parameters
    ----------
    error_type
        type of the error

    molecule_name
        name of the molecule

    errors
        errors between training data and ReaxFF values
    """

    error_string = []

    if "TOTAL AVG" in molecule_name:
        __print_dash()

    for error in errors:
        if error is not None:
            error_string.append(f'{error:10.6f}')
        else:
            error_string.append(f'{" " * 6 }{"None"}')

    one_mol = (f'{error_type.upper()} {molecule_name:20} {error_string[0]} '
               f'{error_string[2]} {error_string[3]} {error_string[4]}')

    print(one_mol)
