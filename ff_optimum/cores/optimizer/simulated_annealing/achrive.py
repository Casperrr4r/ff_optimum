# -*- coding: utf-8 -*-
import copy
from functools import partial
from itertools import filterfalse

import numpy as np

from ff_optimum.cores.compute_builtin import Fitness
from ff_optimum.cores.utilities import (
    EventLogger, file_set_path, get_client, is_client_ready)

__all__ = ['Achrive']

logger = EventLogger(__name__)


class Achrive(object):

    """
    Class for storing solutions

    Attributes
    ----------
    __capacity: int
        soft limit for the solution size

    __solutions: list
        list of Fitness object

    __number_of_save: int
        count of number of save

    Methods
    -------
    capacity
    get_achrive_size
    get_objectives_names
    push_solution
    pop_dominated_solution
    dimension_reduction
    """

    __slots__ = ['__capacity', '__solutions', '__number_of_save']

    def __init__(self, capacity: int=50):

        self.__capacity = capacity

        self.__solutions = list()

        self.__number_of_save = 0

    @property
    def capacity(self) -> int:
        return self.__capacity

    def get_achrive_size(self) -> int:
        return len(self.__solutions)

    def get_objectives_names(self) -> np.ndarray:
        """
        Return a numpy ndarray object containing the objectives names

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            numpy ndarray object containing the objectives names
        """

        return copy.deepcopy(self.__solutions[0]['fitness'].objectives_names)

    def push_solution(self, new_solution: Fitness) -> None:
        """
        if the number of solution in the achrive is larger than the capacity
        pop all the solution dominated by the new solution

        Push the new solution to the achrive

        Parameters
        ----------
        new_solution
            Fitness object of the new solution

        Returns
        -------
        None

        See Also
        --------
        Achrive.pop_dominated_solution

        """

        if len(self.__solutions) > self.__capacity:
            self.pop_dominated_solution(new_solution)

        self.__solutions.append(new_solution)

    def __compute_correlation_matrix(self) -> np.ndarray:
        """
        Compute the correlation matrix of the solution in the achrive

        Parameters
        ----------
        None

        Returns
        -------
        corrcoef
            the correlation matrix
        """

        objective_values_matrix = np.array(
            [solution['fitness'].objectives_values
             for solution in self.__solutions])

        averages = np.tile(np.mean(objective_values_matrix, axis=0),
                           (objective_values_matrix.shape[0], 1))

        shifted_data = objective_values_matrix - averages

        corrcoef = np.corrcoef(shifted_data.T)

        return corrcoef

    @staticmethod
    def __compute_eigen_value_and_vector(matrix: np.ndarray) -> np.ndarray:
        """
        Compute the eigenvalue and eigenvector of the input matrix

        Parameters
        ----------
        matrix
            input matrix

        Returns
        -------
        eigenvalue

        eigenvector
        """

        eigenvalue, eigenvector = np.linalg.eig(matrix)

        return np.real(eigenvalue), np.real(eigenvector)

    def __pca_analysis(self, mode: str) -> np.ndarray:
        """
        Perform principal component analysis to the objectives

        Parameters
        ----------
        mode
            LPCA or PCA

        Returns
        -------
        objectives_names_to_be_poped
            names of the redunent objectives

        See Also
        --------
        Achrive.__compute_correlation_matrix
        Achrive.__compute_eigen_value_and_vector
        """

        # 3 sigma for LPCA, 2 sigma for PCA
        cutoff_cumlative_eigenvalue = 0.997 if "LPCA" in mode else 0.95

        objectives_names_to_be_poped = copy.deepcopy(
            self.__solutions[0]['fitness'].objectives_names)

        logger.info(f'{objectives_names_to_be_poped}')

        corrcoef = self.__compute_correlation_matrix()

        # logger.info(f'Correlation matrix')

        # list(map(lambda correlation: logger.info(f'{correlation}'), corrcoef))

        featValue_real, featVec_real = (
            self.__compute_eigen_value_and_vector(
                corrcoef@corrcoef.T / corrcoef.shape[0]))

        cumulative_eigenvalue = 0

        for index in np.argsort(-featValue_real):

            if cumulative_eigenvalue > cutoff_cumlative_eigenvalue:
                break

            cumulative_eigenvalue += (
                featValue_real[index] / featValue_real.sum())

            logger.info(f'Principal component {index+1}: '
                        f'{featVec_real[index]}')

            logger.info(f'Eigenvalue {featValue_real[index]}')

            component_indices = np.argsort(-featVec_real[index])

            remains = None

            if np.all(featVec_real[index] >= 0):
                remains = np.array([component_indices[0],
                                    component_indices[1]], dtype=np.int)

            elif np.all(featVec_real[index] <= 0):
                remains = np.array([component_indices[-1],
                                    component_indices[-2]], dtype=np.int)
            else:

                if "LPCA" in mode:

                    index1 = component_indices[0]

                    index2 = component_indices[-1]

                    if (featVec_real[index][index1] >=
                            np.abs(featVec_real[index][index2])):

                        remains = np.argwhere(featVec_real[index] > 0)

                        remains = np.append(remains, index2)

                    else:
                        remains = np.argwhere(featVec_real[index] < 0)

                        remains = np.append(remains, index1)

                else:
                    remains = np.array([component_indices[0],
                                        component_indices[-1]], dtype=np.int)

            objectives_names_to_be_poped = np.delete(
                objectives_names_to_be_poped, remains)

        logger.info(f'{objectives_names_to_be_poped}')

        for solution in self.__solutions:
            solution['fitness'].pop_objectives(
                objectives_names_to_be_poped)

        return objectives_names_to_be_poped

    def __reduced_correlation_matrix_analysis(self) -> np.ndarray:
        """
        Perform reduced correlation matrix analysis of objective reduction

        Parameters
        ----------
        None

        Returns
        -------
        objectives_names_to_be_poped_rcm
            array of objectives name to be poped

        See Also
        --------
        Achrive.__compute_correlation_matrix
        Achrive.__compute_eigen_value_and_vector

        """

        objectives_names_to_be_poped_rcm = copy.deepcopy(
            self.__solutions[0]['fitness'].objectives_names)

        reduced_corrcoef = self.__compute_correlation_matrix()

        reduced_correlation_matrix = (
            reduced_corrcoef@reduced_corrcoef.T / reduced_corrcoef.shape[0])

        eiegnvalue, eigenvector = (
            self.__compute_eigen_value_and_vector(reduced_correlation_matrix))

        indices = np.argsort(-eiegnvalue)

        sorted_normalized_featValue = (
            eiegnvalue[indices] / eiegnvalue.sum())

        cut_off = np.cumsum(sorted_normalized_featValue)

        two_sigma = np.argwhere(cut_off < 0.954).size

        m, _ = reduced_correlation_matrix.shape

        correlation_threshold = (1 - eiegnvalue[indices[0]] * (
            1 - two_sigma) / m)

        logger.info(f'Correlation threshold: {correlation_threshold}')

        subsets = []

        # loop over the upper triangle
        # logger.info(f'reduced correlation matrix {reduced_corrcoef}')

        appeared = np.zeros(m, dtype=np.bool)

        for i in range(m):

            sub = []

            for j in range(m):

                if not appeared[j]:

                    indiecs = np.all(
                        np.sign(reduced_corrcoef[i]) ==
                        np.sign(reduced_corrcoef[j]))

                    if i == j or (indiecs and (correlation_threshold <=
                                               reduced_corrcoef[i][j])):
                        sub.append(j)

                        appeared[j] = True

            if sub:
                subsets.append(sub)

            if np.all(appeared):
                break

        for subset in subsets:

            contributions = {}

            max_contribution_index, max_contribution_value = -1, -1

            for index in range(len(subset)):
                contribution_value = np.sum(
                    eiegnvalue * np.abs(eigenvector.T[subset[index]]))

                if contribution_value > max_contribution_value:
                    max_contribution_index = subset[index]

            if max_contribution_index != -1:
                subset.remove(max_contribution_index)

        remove = []

        for subset in subsets:
            remove.extend(subset)

        if subsets and remove:

            objectives_names_to_be_poped_rcm = (
                objectives_names_to_be_poped_rcm[np.unique(remove)])
        else:
            objectives_names_to_be_poped_rcm = None

        return objectives_names_to_be_poped_rcm

    def dimension_reduction(self, mode: str) -> np.ndarray:
        """
        Perform the dimension reduction

        Parameters
        ----------
        mode
            LPCA or PCA

        Returns
        -------
        objectives_names_to_be_poped
            array of objectives name to be poped

        """

        objectives_names_to_be_poped = list()

        objectives_names_to_be_poped_pca = self.__pca_analysis(mode)

        logger.info(f'{objectives_names_to_be_poped_pca}')

        for solution in self.__solutions:
            solution['fitness'].pop_objectives(
                objectives_names_to_be_poped_pca)

        objectives_names_to_be_poped.extend(objectives_names_to_be_poped_pca)

        if "LPCA" in mode:

            objectives_names_to_be_poped_rcm = (
                self.__reduced_correlation_matrix_analysis())

            logger.info(f'{objectives_names_to_be_poped_rcm}')

            if objectives_names_to_be_poped_rcm is not None:

                for solution in self.__solutions:
                    solution['fitness'].pop_objectives(
                        objectives_names_to_be_poped_rcm)

                objectives_names_to_be_poped.extend(
                    objectives_names_to_be_poped_rcm)

        return objectives_names_to_be_poped

    def count_being_dominated_in_achrive(self, fitness: Fitness) -> int:
        """
        See Also
        --------
        count_being_dominated_in_solutions
        """
        return count_being_dominated_in_solutions(self.__solutions, fitness)

    def pop_dominated_solution(self, new_solution: dict) -> None:
        """
        Pop the solution being dominated by the new solution

        Parameters
        ----------
        new_solution
            new solution to be pushed

        Returns
        -------
        None

        See Also
        --------
        cores.compute_builtin.multiobjectives.Fitness.is_dominate
        """

        self.__solutions = list(
            filterfalse(lambda solution: new_solution['fitness'].is_dominate(
                solution['fitness']), self.__solutions))

    def is_inside_achrive(self, new_parameters: dict,
                          package: callable) -> bool:
        """
        Check whether the new set of parameter is inisde the achrive or not

        Parameters
        ----------
        new_parameters
            new parameter set

        package
            user package module

        Returns
        -------
        if the parameter is inside achrive, return True
        Otherwise False
        """

        for solution in self.__solutions:

            if solution.get('parameter', None) is not None:
                if package.is_equal(
                        solution['parameter'], new_parameters):
                    return True

        return False

    def generate_attainment_surface(self, target_size: int=None) -> None:
        """
        Generate the attainmnet surface

        Parameters
        ----------
        target_size
            size of the achrive after finish generating attainment surface

        Returns
        -------
        None

        See Also
        --------
        cores.utilities.parallel_singleton.is_client_ready
        generate_attainment_surface_one
        """

        if self.__solutions:

            if target_size is None:
                target_size = self.__capacity

            required_size = target_size - len(self.__solutions)

            if required_size < 0:
                return

            logger.info(f'Start generating attainment surface required_size: '
                        f'{required_size}')

            number_of_objectives = (
                self.__solutions[0]["fitness"].objectives_values.size)

            objective_names = self.__solutions[0]["fitness"].objectives_names

            objective_values_np = np.array(
                [solution["fitness"].objectives_values
                 for solution in self.__solutions])

            sorted_objective_values_np = np.sort(objective_values_np, axis=0)

            lower = sorted_objective_values_np.T[:, 0]

            upper = sorted_objective_values_np.T[:, -1]

            if is_client_ready():

                partial_f = partial(
                    generate_attainment_surface_one, self.__solutions,
                    objective_names.copy(), sorted_objective_values_np.copy(),
                    lower.copy(), upper.copy())

                direct_view = get_client()[:]

                results = direct_view.map_sync(
                    lambda x: partial_f(), range(required_size))

                self.__solutions.extend(
                    [{"fitness": fitness} for fitness in results])

            else:

                while required_size > 0:

                    fitness = generate_attainment_surface_one(
                        self.__solutions, objective_names,
                        sorted_objective_values_np, lower.copy(), upper.copy())

                    self.__solutions.append({"fitness": fitness})

                    required_size -= 1

            logger.info('Finish generating attainment surface')

    def dump_parameters_to_file(self, package: callable,
                                directory: str) -> None:
        """
        Save all the solution which is not attainment element to file

        Parameters
        ----------
        package
            module of the user package

        directory
            file directory of the parameters to be saved

        Returns
        -------
        None
        """

        count = 0

        directory = file_set_path(directory, f'save_{self.__number_of_save}')

        solutions_copy = [solution for solution in self.__solutions]

        for solution in solutions_copy:
            if solution.get('parameter', None) is not None:
                self.pop_dominated_solution(solution)

        if self.get_achrive_size == 0:
            self.__solutions = solutions_copy

        for solution in self.__solutions:

            parameters = solution.get('parameter', None)

            if parameters is not None:

                file_path = file_set_path(directory, f'ffield_{count}')

                package.save_parameters_to_file(
                    parameters, file_path)

                fitness = solution.get('fitness')

                fitness_path = file_set_path(directory, f'fitness_{count}')

                with open(fitness_path, 'w+') as fp:

                    np.set_printoptions(
                        formatter={'float_kind': lambda x: '%8.6f' % x})

                    for name in fitness.objectives_names:
                        fp.write(f'{name} ')
                    fp.write('\n')
                    fp.write(str(fitness.objectives_values))

                count += 1

                logger.info(f'Save achrive to {file_path} sucess')

        logger.info(f'Number of solution saved: {count}')

        self.__number_of_save += 1


def generate_attainment_surface_one(
        solutions: list, objective_names: np.ndarray,
        sorted_objective_values_np: np.ndarray,
        lower: np.ndarray, upper: np.ndarray, max_iter: int=20) -> Fitness:
    """
    Generate one element of the attainment surface which is the minimum values
    being dominated by one of the solution in the solutions

    Parameters
    ----------
    solutions
        list of Fitness object

    objective_names
        array of objective names

    sorted_objective_values_np
        array of sorted objective values

    lower
        array of minimum objective values

    upper
        array of maximum objective values

    max_iter
        number of maximum time for searching the dominated solution

    Returns
    -------
    fitness object
        which is at least being dominated by one of the solution in the achrive
    """

    temp_objectives_values = np.random.uniform(lower, upper)

    number_of_objectives = temp_objectives_values.size

    for iteration in range(max_iter):

        random_dimensions = np.random.permutation(number_of_objectives)

        for dimension in random_dimensions:

            idx = np.searchsorted(
                sorted_objective_values_np[...,  dimension],
                temp_objectives_values[dimension])

            if idx >= sorted_objective_values_np.shape[1]:
                idx = sorted_objective_values_np.shape[1] - 1

            try:

                temp_objectives_values[idx] = np.take(
                    sorted_objective_values_np[dimension], idx)

            except IndexError:
                return Fitness(objective_names,
                               upper.copy())

            fitness = Fitness(objective_names,
                              temp_objectives_values)

            number_of_being_dominated = (
                count_being_dominated_in_solutions(
                    solutions, fitness))

            if number_of_being_dominated > 0:

                logger.debug('Element generated')

                return fitness

    return Fitness(objective_names, upper.copy())


def count_being_dominated_in_solutions(solutions: list,
                                       fitness: Fitness) -> int:
    """
    Count of the number of solution in the solutions which dominates fitness

    Parameters
    ----------
    solutions
        list of solution to be compared

    fitness
        fitness to be counted

    Returns
    -------
    count: int
        number of solution in the solutions which dominates fitness

    """

    count = 0

    for solution in solutions:
        if solution['fitness'].is_dominate(fitness):
            count += 1

    return count
