# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from ff_optimum.cores.utilities import argument_type_check

__all__ = ['SimulationBox']


class SimulationBox(object):

    """
    Class for simulation box

    Attributes
    ----------

    Methods
    -------
    get_abc_box()
        return the simulation box in abc form
    get_xyz_box()
        return the simulation box in xyz form
    get_volume()
        return the volume of the simulation box

    convert_xml_coordinate_to_lmp_coordinate()
        convert the xml coordinate to LAMMPS coordinate
    """
    __slots__ = ['_a', '_b', '_c',
                 '_alpha', '_beta', '_gamma', '_volume_factor',
                 '_cos_alpha', '_cos_beta', '_cos_gamma',
                 '_sin_gamma', '_tan_gamma',
                 '_lx', '_ly', '_lz', '_xy', '_xz', '_yz']

    @argument_type_check
    def __init__(self, abc_box_list: Union[list, np.ndarray],
                 is_radian: bool=False) -> None:

        self._a = abc_box_list[0]

        self._b = abc_box_list[1]

        self._c = abc_box_list[2]

        self._alpha = \
            (abc_box_list[3] if is_radian else np.deg2rad(abc_box_list[3]))

        self._beta = \
            (abc_box_list[4] if is_radian else np.deg2rad(abc_box_list[4]))

        self._gamma = \
            (abc_box_list[5] if is_radian else np.deg2rad(abc_box_list[5]))

        self._cos_alpha = float()

        self._cos_beta = float()

        self._cos_gamma = float()

        self._sin_gamma = float()

        self._tan_gamma = float()

        self.__compute_sin_cos_tan()

        self._volume_factor = self.__compute_volume_factor()

        self._lx = float()

        self._ly = float()

        self._lz = float()

        self._xy = float()

        self._xz = float()

        self._yz = float()

        self.__convert_abc_box_to_xyz_box()

    def __enter__(self) -> object:
        return self

    def __exit__(self, exc_ty, exc_val, tb) -> None:
        pass

    def __compute_volume_factor(self) -> float:
        """
        Compute the volume factor of the simulation box

        Returns
        -------
        volume factor
        """

        return np.sqrt(1 - self._cos_alpha**2 - self._cos_beta**2 -
                       self._cos_gamma**2 +
                       2 * self._cos_alpha * self._cos_beta * self._cos_gamma)

    def get_abc_box(self) -> np.ndarray:
        """
        Return the simulation box in abc coordinate

        Returns
        -------
        simulation box in abc coordinate
        """

        return np.asarray([self._a, self._b, self._c,
                           self._alpha, self._beta, self._gamma],
                          dtype=np.float128)

    def get_xyz_box(self) -> np.ndarray:
        """
        Return the simulation box in xyz coordinate

        Returns
        -------
        simulation box in xyz coordinate
        """

        return np.asarray([self._lx, self._ly, self._lz,
                           self._xy, self._xz, self._yz], dtype=np.float128)

    def get_volume(self) -> float:
        """
        Return the volume of the simulation box

        Returns
        -------
        volume of the simulation box
        """

        return self._volume_factor * self._a * self._b * self._c

    def convert_xml_coordinate_to_lmp_coordinate(
            self, cartesian_coordinates: list) -> np.ndarray:
        """
        Convert the coordinates in xml to LAMMPS coordinates

        Parameters
        ----------
        cartesian_coordinates

        Returns
        -------
        LAMMPS coordinates
        """

        fractional_coordinates = \
            (self.__convert_cartesian_to_fractional(cartesian_coordinates))

        fractional_coordinates -= np.floor(fractional_coordinates)

        self.__convert_fractional_to_cartesian(fractional_coordinates)

        return self.__convert_fractional_to_cartesian(fractional_coordinates)

    def __compute_sin_cos_tan(self) -> None:
        """
        Compute sine cosine and tangent of alpha beta and gamma

        Returns
        -------
        None

        """

        self._cos_alpha = np.cos(self._alpha)

        self._cos_beta = np.cos(self._beta)

        self._cos_gamma = np.cos(self._gamma)

        self._sin_gamma = np.sin(self._gamma)

        self._tan_gamma = np.tan(self._gamma)

    def __convert_abc_box_to_xyz_box(self) -> None:
        """
        Convert abc simulation box to xyz simulation box

        Returns
        -------
        None

        """

        self._lx = self._a

        self._xy = self._b * self._cos_gamma

        self._xz = self._c * self._cos_beta

        self._ly = np.sqrt(self._b**2 - self._xy**2)

        self._yz = ((self._b * self._c * self._cos_alpha -
                     self._xy * self._xz) / self._ly)

        self._lz = np.sqrt(self._c**2 - self._xz**2 - self._yz**2)

    def __convert_xyz_box_to_abc_box(self) -> None:
        """
        Convert xyz simulation box to abc simulation box

        Returns
        -------
        None

        """

        self._a = self._lx

        self._b = np.hypot(self._ly, self._xy)

        self._c = np.sqrt(self._ly**2 + self._xz**2 + self._yz**2)

        self._alpha = np.arccos((self._xy * self._xz + self._ly * self._yz) /
                                (self._b * self._c))

        self._beta = np.arcos(self._xz / self._c)

        self._gamma = np.arcos(self._xy / self._b)

    def __convert_cartesian_to_fractional(
            self, cartesian_coordinates: np.ndarray) -> np.ndarray:
        """
        Convert cartesian coordinates to cartesian coordinates

        Parameters
        ----------
        cartesian_coordinates
            cartesian coordinates

        Returns
        -------
        fractional_coordinates
            fractional coordinates

        """

        fractional_coordinates = np.zeros(3, dtype=np.float128)

        fractional_coordinates[0] = ((cartesian_coordinates[0] -
                                      cartesian_coordinates[1] /
                                      self._tan_gamma +
                                      cartesian_coordinates[2] *
                                      (self._cos_alpha * self._cos_gamma -
                                       self._cos_beta) /
                                      (self._volume_factor * self._sin_gamma))
                                     / self._a)

        fractional_coordinates[1] = ((cartesian_coordinates[1] /
                                      self._sin_gamma +
                                      cartesian_coordinates[2] *
                                      (self._cos_beta * self._cos_gamma -
                                       self._cos_alpha) /
                                      (self._volume_factor *
                                       self._sin_gamma)) / self._b)

        fractional_coordinates[2] = (cartesian_coordinates[2] /
                                     (self._c * self._volume_factor))

        return fractional_coordinates

    def __convert_fractional_to_cartesian(
            self, fractional_coordinates: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to cartesian coordinates

        Parameters
        ----------
        fractional_coordinates
            fractional coordinates

        Returns
        -------
        cartesian_coordinates
            cartesian coordinates
        """

        cartesian_coordinates = np.zeros(3, dtype=np.float128)

        cartesian_coordinates[0] = (self._a * fractional_coordinates[0] +
                                    self._b * self._cos_gamma *
                                    fractional_coordinates[1] +
                                    self._c * self._cos_beta *
                                    fractional_coordinates[2])

        cartesian_coordinates[1] = (self._b * self._sin_gamma *
                                    fractional_coordinates[1] +
                                    fractional_coordinates[2] * self._c *
                                    (self._cos_alpha - self._cos_beta *
                                     self._cos_gamma) / self._sin_gamma)

        cartesian_coordinates[2] = (fractional_coordinates[2] * self._c *
                                    (self._volume_factor / self._sin_gamma))

        return cartesian_coordinates
