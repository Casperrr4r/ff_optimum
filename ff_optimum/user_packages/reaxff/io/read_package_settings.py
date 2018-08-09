# -*- coding: utf-8 -*-
import re

from ff_optimum.cores.utilities import EventLogger

__all__ = ['read_reaxff_setting']

logger = EventLogger(__name__)


def read_reaxff_setting(setting: dict) -> dict:
    """
    Read ReaxFF package setting from config file

    Returns
    -------
    dictionary object contains the settings

    See Also
    --------
    __read_formation_energy_information_one_mol

    """

    optimization_level = setting.get('optimization_level', 0)

    slient, weights = setting.get('slient', True), setting.get('weights', None)

    composition, dependencies, objectives, mol_weights = {}, {}, {}, {}

    for molecule_name, molecule_info in setting['molecules'].items():
        for objective, function in molecule_info['objectives'].items():

            objective_name = f'{molecule_name.lower()}_{objective}'

            if 'formation' in objective:

                mol_weight = 1.0

                if isinstance(function, list):
                    function, mol_weight = function[0], function[1]

                composition[molecule_name.lower()] = function

                mol_weights[molecule_name.lower()] = mol_weight

                dependencies[molecule_name.lower()], function = (
                    __read_formation_energy_information_one_mol(
                        molecule_name.lower(), function))

            objectives[objective_name] = function

            logger.info(f'Objective: {objective_name}')

    return {"level": optimization_level, "composition": composition,
            "dependencies": dependencies, "objectives": objectives,
            "slient": slient, "weights": weights,
            "mol_weights": mol_weights}


def __read_formation_energy_information_one_mol(
        mol_name: str, combination: str) -> tuple:
    """
    Read formation energy information of one molecule

    Parameters
    ----------
    mol_name
        name of the molecule

    combination
        combination of the formation energy calculation and weights

    Returns
    -------
    dependencies
        other molecule requires for computing the formation energy

    mol_formula
        formula for computing the formation energy

    """

    res = re.match('^\s*(\-*\d+[\.\d+]*)\s*$', combination)

    if res is not None:
        dependencies, mol_formula = None, res.group(1)

    else:

        weights = re.findall('(\d+[\.\d+]*)', combination)

        _dependencies = re.findall(
            '([a-zA-Z]+[a-zA-Z0-9\.]*[\-\w+]*[a-zA-Z]*[a-zA-Z0-9\.]*)',
            combination)

        dependencies = [dependency.lower() for dependency in _dependencies]

        operators = [op.strip()
                     for op in re.findall('\s+[\+|\-]\s+', combination)]

        mol_formula = f'energy["{mol_name}"] - ('

        for name, weight, op in zip(dependencies[:-1], weights[:-1], operators):
            mol_formula += f'{weight} * energy["{name.lower()}"] {op} '
        else:
            mol_formula += (f'{weights[-1]} * energy["{dependencies[-1]}"])')

    return (dependencies, mol_formula)
