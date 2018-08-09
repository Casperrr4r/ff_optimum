# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

__all__ = ['compute_error_builtin']


def compute_error_builtin(calculated_value: np.ndarray,
                          training_data: np.ndarray,
                          arguments: Union[list, str]='rmse') -> float:
    """
    Builtin error function

    Parameters
    ----------
    calculated_value
          numpy.ndarray object storing the calculated value

    training_data
          numpy.ndarray object storing the training data

    arguments
        list or string object contains which error functions will be used for
        calculating the error

    Returns
    -------
    The error

    Raises
    ------
    AttributeError
        If the shape of calculated_value and training_data are different

    See Also
    --------
    builtin_objectives.__compute_mean_absolute_error
    builtin_objectives.__compute_mean_absolute_scaled_error
    builtin_objectives.__compute_root_mean_square_error
    builtin_objectives.__compute_normalized_root_mean_square_error
    """

    if calculated_value.shape[0] != training_data.shape[0]:
        raise AttributeError(
            'Shape of calculated_value and training_data mismatch')

    error_type = arguments if isinstance(arguments, str) else arguments[0]

    err_func = BUILTIN_OBJECTIVES.get(
        error_type, __compute_root_mean_square_error)

    return err_func(calculated_value, training_data)


def __compute_mean_absolute_error(calculated_value: np.ndarray,
                                  training_data: np.ndarray) -> float:
    """
    Compute the mean absolute error between the calculated_value and the
    training_data

    Parameters
    ----------
    calculated_value
        numpy.ndarray object storing the calculated value

    training_data
        numpy.ndarray object storing the training data
    Returns
    -------
    The calculated mean absolute error
    """
    return np.mean(np.abs(np.subtract(calculated_value, training_data)))


def __compute_mean_absolute_scaled_error(calculated_value: np.ndarray,
                                         training_data: np.ndarray) -> float:
    """
    Compute the mean absolute scalederror between
    the calculated_value and the training_data

    Parameters
    ----------
    calculated_value
        numpy.ndarray object storing the calculated value

    training_data
        numpy.ndarray object storing the training data
    Returns
    -------
    The calculated mean absolute scaled error

    See Also
    --------
    builtin_objectives.__compute_mean_absolute_error
    """

    denominator = (np.abs(np.diff(training_data)).sum() /
                   (training_data.shape[0] - 1)
                   if training_data.shape[0] > 1 else 1)

    mase = __compute_mean_absolute_error(calculated_value, training_data)

    if denominator:
        mase /= denominator

    return mase


def __compute_root_mean_square_error(calculated_value: np.ndarray,
                                     training_data: np.ndarray) -> float:
    """
    Compute the root mean square error between the calculated_value and the
    training_data

    Parameters
    ----------
    calculated_value
        numpy.ndarray object storing the calculated value

    training_data
        numpy.ndarray object storing the training data
    Returns
    -------
    The calculated root mean square error
    """

    return np.sqrt(np.mean(np.square(np.subtract(calculated_value,
                                                 training_data))))


def __compute_normalized_root_mean_square_error(
        calculated_value: np.ndarray, training_data: np.ndarray) -> float:
    """
    Compute the normalized root mean square error between
    the calculated_value and the training_data

    Parameters
    ----------
    calculated_value
        numpy.ndarray object storing the calculated value

    training_data
        numpy.ndarray object storing the training data
    Returns
    -------
    The calculated nomralized root mean square error

    See Also
    --------
    builtin_objectives.__compute_root_mean_square_error
    """

    nrmse = __compute_root_mean_square_error(calculated_value, training_data)

    range = training_data.max() - training_data.min()

    if range:
        nrmse /= range

    return nrmse

BUILTIN_OBJECTIVES = {'mae': __compute_mean_absolute_error,
                      'mase': __compute_mean_absolute_scaled_error,
                      'rmse': __compute_root_mean_square_error,
                      'nrmse': __compute_normalized_root_mean_square_error}
