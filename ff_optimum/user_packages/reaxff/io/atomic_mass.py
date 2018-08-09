# -*- coding: utf-8 -*-
from enum import Enum, unique

__all__ = ['ElementNotFoundError', 'get_element_name_from_atomic_mass']


class ElementNotFoundError(Exception):

    """ raise when element is not found """
    pass


@unique
class AtomicMass(float, Enum):

    """
    Enumerate Class for mapping element to its atomic mass
    """

    H = 1.0078250
    Li = 6.941
    C = 12
    N = 14
    O = 15.9949146
    Si = 27.9769284
    P = 30.973762
    S = 32.065
    Cu = 62.9295992
    Ge = 73.9211788
    Mo = 97.9055000
    Te = 129.9067000
    X = 131


def get_element_name_from_atomic_mass(_mass: str) -> str:
    """
    Get element string form the enumerate class by mass string

    Parameters
    ----------
    _mass
        mass in string

    Returns
    -------
    atomic_masses[mid].name
        name of the element

    Raises
    ------
    ElementNotFoundError

    """

    mass = float(_mass)

    atomic_masses = list(AtomicMass)

    start = mid = 0

    end = len(atomic_masses) - 1

    while start <= end:

        mid = int((end + start) / 2)

        if abs(atomic_masses[mid].value - mass) < 0.5:
            return atomic_masses[mid].name

        if atomic_masses[mid].value > mass:
            end = mid - 1
        else:
            start = mid + 1

    raise ElementNotFoundError()

    return None
