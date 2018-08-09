# -*- coding: utf-8 -*-
import numpy as np

try:
    import matplotlib.pyplot as plt
    PLT_INSTALLED = True
except ImportError:
    PLT_INSTALLED = False

from ff_optimum.cores.utilities.file_path import (
    file_set_path, file_create_directory)


__all__ = ['plot_reaxff_results']


SCANTYPE = {'EOS': "Volume/at (A3)", 'ATOM': "Volume/at (A3)",
            'NEB': "Reaction coordinate", 'DISS': "Distance (A)",
            'ANG': "Angle (degres)", 'TOR': "Angle (degres)"}


def plot_reaxff_results(package_settings: dict, plot_information: dict,
                        calculated_values: dict, training_data: dict,
                        output_directory: str) -> None:
    """
    Plot the calculated results of ReaxFF

    Parameters
    ----------

    package_settings
        package settings

    plot_information
        scantype and the x axis values

    calculated_values
        ReaxFF computed values

    training_data
        DFT training data

    output_directory
        directory for saving the plot

    Returns
    -------
    None
    """

    objectives = package_settings.get('objectives', None)

    file_create_directory(output_directory)

    if objectives is not None:

        for objective in objectives:

            mol_name = objective.split('_')[0]

            if 'charge' in objective:
                __plot_charge(mol_name, plot_information, calculated_values,
                              training_data, output_directory)
            elif 'energy' in objective:
                __plot_energy(mol_name, plot_information, calculated_values,
                              training_data, output_directory)


def __plot_energy(mol_name: str, plot_information: dict,
                  calculated_values: list, training_data: list,
                  output_directory: str) -> None:
    """
    Plot energy according to scantype and save it to the output directory

    Parameters
    ----------

    mol_name
        name of the molecule

    plot_information
        scantype and the x axis values

    calculated_values
        ReaxFF computed values

    training_data
        DFT training data

    output_directory
        directory for saving the plot

    Returns
    -------
    None
    """

    if not PLT_INSTALLED:
        logger.error('matplotlib is not found')
        return

    label = SCANTYPE.get(plot_information[mol_name]['scantype'])

    target_energies = np.zeros(len(calculated_values[mol_name]))

    calculated_eneregies = np.zeros_like(target_energies)

    for idx in range(target_energies.size):

        np.put(target_energies, idx,
               training_data[mol_name][idx]['energy'])

        np.put(calculated_eneregies, idx,
               calculated_values[mol_name][idx]['energy'])

    min_idx = np.argmin(target_energies)

    target_energies -= np.take(target_energies, min_idx)

    calculated_eneregies -= np.take(calculated_eneregies, min_idx)

    x = np.array(plot_information[mol_name]['values'])

    indices = np.argsort(x)

    plt.plot(x[indices], target_energies[indices],
             marker='o', linestyle='-', label='DFT', color="r")

    plt.plot(x[indices], calculated_eneregies[indices],
             marker='*', linestyle='--', label='ReaxFF', color="b",
             alpha=0.5)

    plt.xlabel(label, fontsize=14)

    plt.ylabel('E (kcal/mol)', fontsize=14)

    plt.legend(loc=1)

    plt.title(mol_name, fontsize=14)

    plt.savefig(file_set_path(output_directory, f'{mol_name}-energy.png'))

    plt.gcf().clear()


def __plot_charge(mol_name: str, plot_information: dict,
                  calculated_values: list, training_data: list,
                  output_directory: str) -> None:
    """
    Plot charge according to scantype

    Parameters
    ----------

    mol_name
        name of the molecule

    plot_information
        scantype and the x axis values

    calculated_values
        ReaxFF computed values

    training_data
        DFT training data

    output_directory
        directory for saving the plot
    """

    if not PLT_INSTALLED:
        logger.error('matplotlib is not found')
        return

    label = SCANTYPE.get(plot_information[mol_name]['scantype'])

    x = np.array(plot_information[mol_name]['values'])

    colormap = plt.cm.rainbow

    natom = len(calculated_values[mol_name][0]['q'])

    cm = [colormap(i) for i in np.linspace(0, 0.9, natom)]

    for idx in range(len(calculated_values[mol_name]) - 1):

        xx = np.zeros(natom)

        xx.fill(x[idx])

        plt.scatter(xx, training_data[mol_name][idx]['q'],
                    marker='o', lw=0, color=cm)

        plt.scatter(xx, calculated_values[mol_name][idx]['q'],
                    marker='*', lw=0, color=cm)

    plt.scatter(xx, training_data[mol_name][-1]['q'],
                marker='o', lw=0, label='DFT', color=cm)

    plt.scatter(xx, calculated_values[mol_name][-1]['q'],
                marker='*', lw=0, label='ReaxFF', color=cm)

    plt.xlabel(label, fontsize=14)

    plt.ylabel('Q (e)', fontsize=14)

    plt.legend(loc=1)

    plt.title(mol_name, fontsize=14)

    plt.savefig(file_set_path(output_directory, f'{mol_name}-charge.png'))

    plt.gcf().clear()
