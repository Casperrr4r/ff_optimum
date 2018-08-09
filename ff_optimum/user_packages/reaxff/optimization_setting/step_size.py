# -*- coding: utf-8 -*-
import copy

import numpy as np

from .parameters import REAXFF_PARAM_DTYPE

__all__ = ['get_default_step_size']


def get_default_step_size(optimization_level: int=0) -> dict:
    """
    Get the default step size by the optimization level

    Parameters
    ----------
    optimization_level
        optimization level range from 0 to 5
        where level 0 is the QEq parameters

    Returns
    -------
    step_size: np.ndarray
        np.ndarray object of the step size

    See Also
    --------
    __get_default_step_size_reaxff_helper

    """

    category_by_level = __get_step_size_by_optimization_level(
        optimization_level)

    pmove = map(lambda category:
                __get_default_step_size_reaxff_helper(category),
                category_by_level.values())

    step_size = dict(zip(REAXFF_PMOVE_CATEGORY.keys(), pmove))

    return step_size


def __get_default_step_size_reaxff_helper(pmove: dict) -> np.ndarray:
    """
    Helper function for packing the step_size from dictionary to np.ndarray

    Parameters
    ----------
    pmove

    Returns
    -------
    np.ndarray

    """

    return np.fromiter(zip(pmove.keys(), pmove.values()),
                       dtype=REAXFF_PARAM_DTYPE)


def __get_step_size_by_optimization_level(
        optimization_level: int=0) -> dict:
    """

    Parameters
    ----------
    optimization_level
        level of the optimization range from 0 to 5

    Returns
    -------
    category_by_level
        a dictionary object of step size

    See Also
    --------
    __get_step_sizes_by_optimization_level_helper

    """

    if not isinstance(optimization_level, int):
        optimization_level = int(optimization_level)

    category_by_level = copy.deepcopy(REAXFF_PMOVE_CATEGORY)

    if optimization_level == 0:

        atom_param_to_be_optimized = ['gammaEEM', 'chiEEM', 'etaEEM']

        __get_step_sizes_by_optimization_level_helper(
            category_by_level['atoms'], atom_param_to_be_optimized)

        for key in filter(lambda key: key not in 'atoms',
                          category_by_level.keys()):

            __get_step_sizes_by_optimization_level_helper(
                category_by_level[key], [])

    elif optimization_level == 1 or optimization_level == 2:

        __get_step_sizes_by_optimization_level_helper(
            category_by_level['atoms'],
            ['cov.r', 'Rvdw', 'Evdw', 'alfa', 'gammavdW'])

        __get_step_sizes_by_optimization_level_helper(
            category_by_level['bonds'], ['Edis1', 'Edis2', 'Edis3'])

        if optimization_level == 1:
            for key in filter(lambda key: key not in ['atoms', 'bonds'],
                              category_by_level.keys()):

                __get_step_sizes_by_optimization_level_helper(
                    category_by_level[key], [])

        else:
            __get_step_sizes_by_optimization_level_helper(
                category_by_level['off-diagonal'],
                ['Ediss', 'Ro', 'gamma'])

            __get_step_sizes_by_optimization_level_helper(
                category_by_level['angles'], ['Thetao', 'ka', 'kb'])

            for key in filter(lambda key: key in ['torsions', 'hydrogen'],
                              category_by_level.keys()):

                __get_step_sizes_by_optimization_level_helper(
                    category_by_level[key], [])

    elif optimization_level >= 3:

        atom_param_to_be_optimized = [
            'cov.r', 'Rvdw', 'Evdw', 'cov.r2', 'cov.r3']

        if optimization_level >= 4:

            atom_param_to_be_optimized.extend([
                'Eunder', 'Eover', 'Elp', 'ovun',
                '13BO1', '13BO2', '13BO3', 'val1',
                'val3', 'val4'])

            if optimization_level >= 5:
                atom_param_to_be_optimized.extend(['alfa', 'gammavdW'])

        __get_step_sizes_by_optimization_level_helper(
            category_by_level['atoms'], atom_param_to_be_optimized)

        __get_step_sizes_by_optimization_level_helper(
            category_by_level['bonds'],
            ['Edis1', 'Edis2', 'Edis3', 'pbe1',
             'pbo5', '13corr', 'pbo6', 'povon1',
             'pbe2', 'pbo3', 'pbo4', 'pbo1', 'pbo2'])

        __get_step_sizes_by_optimization_level_helper(
            category_by_level['off-diagonal'],
            ['Ediss', 'Ro', 'gamma', 'rsigma', 'rpi', 'rpi2'])

        __get_step_sizes_by_optimization_level_helper(
            category_by_level['angles'],
            ['Thetao', 'ka', 'kb', 'pv1', 'pv2', 'valbo', 'pv3'])

        __get_step_sizes_by_optimization_level_helper(
            category_by_level['torsions'],
            ['V1', 'V2', 'V3', 'V2BO', 'vconj'])

        __get_step_sizes_by_optimization_level_helper(
            category_by_level['hydrogen'],
            ['Rhb', 'Dehb', 'vhb1', 'vhb2'])

    return category_by_level


def __get_step_sizes_by_optimization_level_helper(
        category: dict, parameter_to_be_optimized: list) -> dict:
    """
    Set the step size of the parameters not to be optimized to zero

    Parameters
    ----------

    category
        step_size of one category of parameters

    parameter_to_be_optimized
        list of names of the parameter to be optimized

    Returns
    -------
    category
    """

    for key in filter(lambda key: key not in parameter_to_be_optimized,
                      category.keys()):

        category[key] = 0.0

    return category


REAXFF_TYPE_PMOVE = {
    'cov.r': 0.005, 'valency': 0.0, 'a.m': 0.0, 'Rvdw': 0.01,
    'Evdw': 0.01, 'gammaEEM': 0.01, 'cov.r2': 0.01, '#el.': 0.0,
    'alfa': 0.1, 'gammavdW': 0.1, 'valency2': 0.0, 'Eunder': 3.0,
    'Eover': 3.0, 'chiEEM': 0.01, 'etaEEM': 0.01, 'n.u.1': 0.0,
    'cov.r3': 0.01, 'Elp': 0.1, 'Heatinc.': 0.0, '13BO1': 0.1,
    '13BO2': 0.1, '13BO3': 0.1, 'n.u.2': 0.0, 'n.u.3': 0.0,
    'ovun': 0.5, 'val1': 0.01, 'n.u.4': 0.0, 'val3': 0.01,
    'val4': 0.01, 'n.u.5': 0.0, 'n.u.6': 0.0, 'n.u.7': 0.0}

REAXFF_BOND_PMOVE = {
    'Edis1': 0.01, 'Edis2': 0.1, 'Edis3': 0.01, 'pbe1': 0.01,
    'pbo5': 0.01, '13corr': 0.0, 'pbo6': 0.01, 'povon1': 0.01,
    'pbe2': 0.5, 'pbo3': 0.01, 'pbo4': 0.01, 'n.u.1': 0.0,
    'pbo1': 0.01, 'pbo2': 0.01, 'ovcorr': 0.0}

REAXFF_DIAG_PMOVE = {
    'Ediss': 0.01, 'Ro': 0.01, 'gamma': 0.01, 'rsigma': 0.01,
    'rpi': 0.01, 'rpi2': 0.01}

REAXFF_ANGLE_PMOVE = {
    'Thetao': 0.01, 'ka': 0.1, 'kb': 0.1, 'pv1': 0.1,
    'pv2': 0.1, 'valbo': 0.1, 'pv3': 0.01}

REAXFF_TORSIONS_PMOVE = {
    'V1': 0.1, 'V2': 0.1, 'V3': 0.1, 'V2BO': 0.01, 'vconj': 0.01,
    'n.u.1': 0.0, 'n.u.2': 0.0}

REAXFF_HYDROGEN_BOND_PMOVE = {
    'Rhb': 0.01, 'Dehb': 0.01, 'vhb1': 0.01, 'vhb2': 0.0}


REAXFF_PMOVE_CATEGORY = {
    'atoms': REAXFF_TYPE_PMOVE,
    'bonds': REAXFF_BOND_PMOVE,
    'off-diagonal': REAXFF_DIAG_PMOVE,
    'angles': REAXFF_ANGLE_PMOVE,
    'torsions': REAXFF_TORSIONS_PMOVE,
    'hydrogen': REAXFF_HYDROGEN_BOND_PMOVE}
