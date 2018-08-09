# -*- coding: utf-8 -*-
from typing import Generator

import numpy as np

from ff_optimum.cores.utilities import EventLogger


__all__ = ['REAXFF_NUMBER_OF_STRESS', 'REAXFF_NUMBER_OF_ENERGY',
           'REAXFF_PARAM_DTYPE', 'REAXFF_GENERAL_PARAMS',
           'REAXFF_TYPE_PARAMS', 'REAXFF_BOND_PARAMS',
           'REAXFF_DIAG_PARAMS', 'REAXFF_ANGLE_PARAMS',
           'REAXFF_TORSION_PARAMS', 'REAXFF_HYDROGEN_BOND_PARAMS',
           'REAXFF_PARAMETER_CATEGORY', 'next_parameter_generator', 'is_equal']

REAXFF_NUMBER_OF_STRESS = 9

REAXFF_NUMBER_OF_ENERGY = 1

REAXFF_PARAM_DTYPE = [('name', 'U12'), ('value', 'f8')]

REAXFF_GENERAL_PARAMS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6',
                         'G7', 'G8', 'G9', 'G10', 'G11', 'G12',
                         'G13', 'G14', 'G15', 'G16', 'G17', 'G18',
                         'G19', 'G20', 'G21', 'G22', 'G23', 'G24',
                         'G25', 'G26', 'G27', 'G28', 'G29', 'G30',
                         'G31', 'G32', 'G33', 'G34', 'G35', 'G36',
                         'G37', 'G38', 'G39']

REAXFF_TYPE_PARAMS = ['cov.r', 'valency', 'a.m', 'Rvdw',
                      'Evdw', 'gammaEEM', 'cov.r2', '#el.',
                      'alfa', 'gammavdW', 'valency2', 'Eunder',
                      'Eover', 'chiEEM', 'etaEEM', 'n.u.1',
                      'cov.r3', 'Elp', 'Heatinc.', '13BO1',
                      '13BO2', '13BO3', 'n.u.2', 'n.u.3',
                      'ovun', 'val1', 'n.u.4', 'val3',
                      'val4', 'n.u.5', 'n.u.6', 'n.u.7']

REAXFF_BOND_PARAMS = ['Edis1', 'Edis2', 'Edis3', 'pbe1',
                      'pbo5', '13corr', 'pbo6', 'povon1',
                      'pbe2', 'pbo3', 'pbo4', 'n.u.1',
                      'pbo1', 'pbo2', 'ovcorr']

REAXFF_DIAG_PARAMS = ['Ediss', 'Ro', 'gamma', 'rsigma', 'rpi', 'rpi2']

REAXFF_ANGLE_PARAMS = ['Thetao', 'ka', 'kb', 'pv1', 'pv2',
                       'valbo', 'pv3']

REAXFF_TORSION_PARAMS = ['V1', 'V2', 'V3', 'V2BO', 'vconj',
                         'n.u.1', 'n.u.2']

REAXFF_HYDROGEN_BOND_PARAMS = ['Rhb', 'Dehb', 'vhb1', 'vhb2']

REAXFF_PARAMETER_CATEGORY = {
    'atoms': REAXFF_TYPE_PARAMS,
    'bonds': REAXFF_BOND_PARAMS,
    'off-diagonal': REAXFF_DIAG_PARAMS,
    'angles': REAXFF_ANGLE_PARAMS,
    'torsions': REAXFF_TORSION_PARAMS,
    'hydrogen': REAXFF_HYDROGEN_BOND_PARAMS}

logger = EventLogger(__name__)


def is_equal(parameters_a: dict, parameters_b: dict) -> bool:
    """
    Check whether two set of parameters values is equal or not
    with a tolrence 1e-5

    Parameters
    ----------
    parameters_a
        the first set of parameters

    parameters_b
        the second set of parameters

    Returns
    -------
    True if both of them are equal
    False otherwise

    """

    for category_a, category_b in zip(parameters_a.values(),
                                      parameters_b.values()):
        for values_a, values_b in zip(category_a.values(), category_b.values()):
            if not isinstance(values_a, float):
                if not np.isclose(values_a['value'], values_b['value'],
                                  atol=1e-5).all():
                    return False

    return True


def next_parameter_generator(parameters: dict,
                             step_size: dict,
                             constraints: dict,
                             slient: bool=False) -> Generator:
    """
    Generator generating the parameter to be optimized

    Parameters
    ----------
    parameters
        the parameter to be optimized

    sensitivities
        the information about the maximum change of parameter values

    constraints
        the constraints of the paramters

    Yields
    ------
    idx
        the index of the parameter to be perturbed

    values['value']
        the array object containing the parameters value

    step_size[category_name]['value'][idx]
        the maximum step_size of the parameter to be moved

    constraints[category_name]['lower_bound'][idx]
        the lower bound of the parameter values

    constraints[category_name]['upper_bound'][idx]
        the upper bound of the parameter values
    """

    for category_name, category in parameters.items():
        for key, values in category.items():

            if not isinstance(values, float):
                for idx in range(values.size):
                    if step_size[category_name]['value'][idx]:
                        yield (idx, values['value'],
                               step_size[category_name]['value'][idx],
                               (constraints[category_name]['lower_bound'][idx],
                                constraints[category_name]['upper_bound'][idx])
                               )
