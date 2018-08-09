# -*- coding: utf-8 -*-
from ff_optimum.cores.utilities import ConfigReader

from .simulated_annealing import (
    DominanceBasedMultiobjectiveSimulatedAnnealingOptimizer,
    SimulatedAnnealingOptimizer)


__all__ = ['OptimizerFactory']


class OptimizerFactory(object):

    """
    Class for creating optimizer object

    Methods
    -------
    create_optimizer_from_config(config_path)
    """

    @classmethod
    def create_optimizer_from_config(cls, config_path: str) -> object:
        """
        Parameters
        ----------
        config_path
              string contain the file path of the config file

        Returns
        -------
        Optimizer
              Optimizer object base on the config

        Raises
        ValueError
              If the algorithm is not found

        See Also
        --------
        config_reader.config_reader.ConfigReader
        """

        reader = ConfigReader(config_path)

        mosa = 'dominance_based_multiobjective_simulated_annealing'

        if mosa in reader.algorithm_name:
            return DominanceBasedMultiobjectiveSimulatedAnnealingOptimizer(
                reader.number_of_processors, reader.profile,
                reader.package_name, reader.package_settings,
                reader.param_initial_values,
                reader.constraints_source, reader.constraints_input,
                reader.commands_holder_train, reader.training_data,
                reader.plot_information, reader.alogrithm_parameters,
                reader.output_directory)

        elif 'simulated_annealing' in reader.algorithm_name:
            return SimulatedAnnealingOptimizer(
                reader.number_of_processors, reader.profile,
                reader.package_name, reader.package_settings,
                reader.param_initial_values,
                reader.constraints_source, reader.constraints_input,
                reader.commands_holder_train, reader.training_data,
                reader.plot_information, reader.alogrithm_parameters,
                reader.output_directory)

        else:
            raise ValueError('No such algorithm')
