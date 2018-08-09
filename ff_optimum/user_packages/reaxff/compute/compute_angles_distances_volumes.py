# -*- coding: utf-8 -*-
import numpy as np

__all__ = ['compute_bond_angle', 'compute_bond_distance',
           'compute_volume', 'compute_torsion_angle']


def compute_bond_distance(position1: np.ndarray,
                          position2: np.ndarray) -> float:
    """
    Compute bond distance of a two body bond

    Parameters
    ----------
    position1
        position of the first atom

    position2
        position of the second atom

    Returns
    -------
    float
        the bond distance
    """

    return np.sqrt(np.sum((position1 - position2)**2))


def compute_bond_angle(position1: np.ndarray, position2: np.ndarray,
                       position3: np.ndarray) -> float:
    """
    Compute bond angle of a three body bond

    Parameters
    ----------
    position1
        position of the first atom

    position2
        position of the second atom

    position3
        position of the third atom

    Returns
    -------
    float
        the bond angle
    """

    position2_1 = poistion2 - position1

    position3_2 = position3 - position2

    bond_angle = (np.arccos(
        np.dot(position2_1, position3_2) /
        np.real(np.sqrt(np.dot(position2_1, position2_1) *
                        np.dot(position3_2, position3_2)))))

    return np.rad2deg(bond_angle)


def compute_volume(box: np.ndarray, is_radian: bool=False) -> float:
    """
    Compute bond angle of a three body bond

    Parameters
    ----------
    box
        position of the simulation box

    is_radian
        whether the angle in box is radian or not

    Returns
    -------
    float
        the volume
    """

    angles = box[3:]

    if not is_radian:
        angles = np.deg2rad(angles)

    cos_angles = np.cos(angles)

    return (np.prod(box[0:3]) *
            np.sqrt(1 + 2 * np.prod(cos_angles) - (cos_angles**2).sum()))


def compute_torsion_angle(position1: np.ndarray,
                          position2: np.ndarray,
                          position3: np.ndarray,
                          position4: np.ndarray) -> float:
    """
    Compute bond angle of a three body bond

    Parameters
    ----------
    position1
        position of the first atom

    position2
        position of the second atom

    position3
        position of the third atom

    position4
        position of the 4th atom

    Returns
    -------
    float
        the bond angle
    """

    position1_2 = position2 - position1

    position2_3 = position3 - position2

    position3_4 = position4 - position3

    rn1 = np.cross(position1_2, position2_3)

    rn2 = np.cross(position2_3, position3_4)

    r = np.cross(position1_2, position3_4)

    n = np.dot(rn1, rn2)

    n1 = np.real(np.sqrt(numpy.dot(rn1, rn1)))

    n2 = np.real(np.sqrt(numpy.dot(rn2, rn2)))

    torsion_angle = np.rad2deg(np.arccos(n / (n1 * n2)))

    if r[2] >= 0:
        return torsion_angle

    return (360 - torsion_angle)
