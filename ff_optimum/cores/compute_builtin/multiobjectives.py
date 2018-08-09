# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from ff_optimum.cores.utilities.event_logging import EventLogger

__all__ = ['Fitness']

logger = EventLogger(__name__)


class Fitness(object):

    """
    Class for storing multi-objective fitness

    Attributes
    ----------
    __objectives_names: np.ndarray
        names of objectives

    __objectives_values: np.ndarray
        values of objectives

    Methods
    -------
    pop_objectives(objectives_names)
        pop objective from the object by names

    is_dominate(fitness)
        return true if the fitness dominate the input fitness
    """

    def __init__(self, objectives_names: list,
                 objectives_values: np.ndarray) -> None:

        self.__objectives_names = np.array(list(objectives_names))

        self.__objectives_values = objectives_values

    @property
    def objectives_names(self) -> list:
        return self.__objectives_names

    def pop_objectives(self, objectives_names: Union[list, np.ndarray]) -> None:
        """
        Pop objectives by the objectives_names

        Parameters
        ----------
        objectives_names
            list or np.array object containing name of objective to be poped

        Returns
        -------
        None

        """

        mask = [np.argwhere(self.__objectives_names == objectives_name)
                for objectives_name in objectives_names]

        self.__objectives_names = np.delete(self.__objectives_names, mask)

        self.__objectives_values = np.delete(self.__objectives_values, mask)

    @property
    def objectives_values(self) -> np.ndarray:
        return self.__objectives_values

    def is_dominate(self, fitness: object) -> bool:
        """
        Check whether if the fitness dominate the input fitness

        Parameters
        ----------
        fitness
            Fitness object

        Returns
        -------
        True if the fitness in this object dominate the input fitness
        Otherwise False
        """

        if np.core.defchararray.equal(self.__objectives_names,
                                      fitness.objectives_names).all():

            rounded_obj_values = np.around(
                self.__objectives_values, decimals=8)

            rounded_fitness = np.around(fitness.objectives_values, decimals=8)

            cond1 = np.less(rounded_obj_values,
                            rounded_fitness)

            cond2 = np.less_equal(rounded_obj_values, rounded_fitness)

            cond = np.logical_or(cond1, cond2)

            if cond.all() and cond1.any():
                return True

            return False

        else:
            raise ValueError('objectives_names is unmatched')
