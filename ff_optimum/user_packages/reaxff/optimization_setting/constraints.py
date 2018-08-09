# -*- coding: utf-8 -*-
import numpy as np

from .parameters import REAXFF_PARAMETER_CATEGORY

__all__ = ['get_default_constraints']


def get_default_constraints() -> dict:
    """
    Get the default constraints of ReaxFF parameters

    Returns
    -------
    constraints

    See Also
    --------
    __get_default_constraints_helper

    """

    constraints = {}

    packed_constraints = __get_default_constraints_helper()

    for category, param_name in REAXFF_PARAMETER_CATEGORY.items():

        constraints[category] = (
            np.fromiter(zip(param_name,
                            np.zeros(len(param_name), dtype='f8'),
                            np.zeros(len(param_name), dtype='f8')),
                        dtype=REAXFF_CONSTRAINTS_DTYPE))

        for idx in range(constraints[category].size):

            packed_idx = np.where(
                packed_constraints['name'] ==
                    constraints[category]['name'][idx])

            np.put(constraints[category], idx, packed_constraints[packed_idx])

    return constraints


def __get_default_constraints_helper() -> np.ndarray:
    """
    Pack the default constraints from REAXFF_CONSTRAINTS

    Returns
    -------
    np,ndarray
    """

    lower_bound = (value[0] for value in REAXFF_CONSTRAINTS.values())

    upper_bound = (value[1] for value in REAXFF_CONSTRAINTS.values())

    return np.fromiter(
        zip(REAXFF_CONSTRAINTS.keys(), lower_bound, upper_bound),
            dtype=REAXFF_CONSTRAINTS_DTYPE)


REAXFF_CONSTRAINTS_DTYPE = [('name', 'U12'),
                            ('lower_bound', 'f8'), ('upper_bound', 'f8')]

REAXFF_CONSTRAINTS = {
    'cov.r': [0.655500, 3.090400],
    'valency': [1.000000, 6.000000],
    'a.m': [1.008000, 127.600400],
    'Rvdw': [0.700000, 2.900000],
    'Evdw': [0.022800, 3.865400],
    'gammaEEM': [0.200000, 1.200000],
    'cov.r2': [-1.683600, 1.562900],
    '#el.': [1.000000, 7.000000],
    'alfa': [8.223000, 14.607300],
    'gammavdW': [1.000000, 100.000000],
    'valency2': [1.000000, 5.000000],
    'Eunder': [0.000000, 52.999800],
    'Eover': [0.000000, 139.930900],
    'chiEEM': [-1.364700, 10.000000],
    'etaEEM': [5.000000, 15.000000],
    'n.u.': [0.000000, 2.000000],
    'cov.r3': [-1.300000, 1.460100],
    'Elp': [0.000000, 35.000000],
    'Heatinc.': [-2.370000, 206.791000],
    '13BO1': [-23.723100, 100.000000],
    '13BO2': [-0.075300, 34.928900],
    '13BO3': [0.000000, 15.142500],
    'n.u.1': [0.000000, 1.069800],
    'n.u.2': [0.000000, 0.000000],
    'ovun': [-40.000000, -1.000000],
    'val1': [0.300000, 10.000000],
    'n.u.3': [1.000000, 1.056400],
    'val3': [1.000000, 12.000000],
    'val4': [2.263200, 3.641100],
    'n.u.4': [0.000000, 1.400000],
    'n.u.5': [0.000000, 0.100000],
    'n.u.6': [0.000000, 12.000000],
    'Edis1': [0.000000, 300.000000],
    'Edis2': [0.000000, 220.000000],
    'Edis3': [0.000000, 167.613200],
    'pbe1': [-1.000000, 1.000000],
    'pbo5': [-0.555800, 0.300000],
    '13corr': [0.000000, 1.000000],
    'pbo6': [6.000000, 49.561100],
    'povon1': [0.002500, 1.250000],
    'pbe2': [-0.451000, 10.000000],
    'pbo3': [-1.000000, 1.000000],
    'pbo4': [0.000000, 30.000000],
    'n.u.7': [0.000000, 1.000000],
    'pbo1': [-0.519100, -0.003000],
    'pbo2': [2.000000, 10.000000],
    'ovcorr': [0.000000, 1.000000],
    'Ediss': [0.010000, 9.835500],
    'Ro': [1.288500, 2.350000],
    'gamma': [5.777000, 13.500000],
    'rsigma': [-1.000000, 3.000000],
    'rpi': [-1.000000, 1.714300],
    'rpi2': [-1.000000, 1.645900],
    'Thetao': [0.000000, 103.320400],
    'ka': [0.000000, 66.782500],
    'kb': [0.010000, 12.000000],
    'pv1': [-38.420000, 2.499600],
    'pv2': [-0.655300, 4.622800],
    'valbo': [-50.000000, 68.107200],
    'pv3': [1.001000, 4.665000],
    'V1': [-6.000000, 4.000000],
    'V2': [-42.773800, 150.000000],
    'V3': [-1.500000, 1.916500],
    'V2BO': [-11.527400, 0.000000],
    'vconj': [-3.000000, 0.000000],
    'n.u.8': [0.000000, 0.000000],
    'n.u.9': [0.000000, 0.000000],
    'Rhb': [1.500000, 4.047600],
    'Dehb': [-9.840700, 0.000000],
    'vhb1': [1.450000, 4.907600],
    'vhb2': [4.235700, 23.000000]}
