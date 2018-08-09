# -*- coding: utf-8 -*-
from enum import Enum, unique

from ff_optimum.cores.utilities import EventLogger

__all__ = ['ReaxFFObjectives', 'pop_pressure_from_objectives',
           'pop_objective_from_settings']

logger = EventLogger(__name__)


@unique
class ReaxFFObjectives(Enum):

    """
    Enum class for the ReaxFF objectives
    """

    energy = 0
    formation = 1
    charge = 2
    force = 3
    stress = 4


def pop_pressure_from_objectives(package_settings: dict) -> None:
    """
    Pop all pressure objectives from the package settings

    Parameters
    ----------
    package_settings

    Returns
    -------
    None

    """

    targets = []

    objectives = package_settings['objectives']

    for objective_name in objectives.keys():
        if objective_name.endswith('stress'):
            targets.append(objective_name)

    for target in targets:
        objectives.pop(target, None)
        logger.info(f'Objective: {target} is poped')


def pop_objective_from_settings(package_settings: dict,
                                objective_name: str) -> None:
    """
    Pop one objective from package_settings by the objective name

    Parameters
    ----------
    package_settings
        package settings

    objective_name
        name of the objective
    """

    objectives = package_settings['objectives']

    pop = True

    # if objective_name.endswith('energy'):

    #    dependencies = package_settings['dependencies']

    #    mol_name = objective_name.split('_energy')[0]

    #    for dependency in dependencies:
    #        if mol_name in dependency:
    #            pop = False

    if objective_name.endswith('formation'):

        dependencies = package_settings['dependencies']

        mol_name = objective_name.split('_energy')[0]

        for dependency in dependencies:
            if mol_name in dependency:
                pop = False

    if pop:

        objectives.pop(objective_name, None)

        logger.info(f'Objective: {objective_name} is poped')

        return True

    else:
        objectives.pop(objective_name, None)

        return False
