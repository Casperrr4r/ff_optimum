# -*- coding: utf-8 -*-
from functools import wraps
from typing import Any, Callable

__all__ = ['argument_type_check']


def argument_type_check(function: Callable) -> Callable:
    """
    Decorator to check whether the type of input arguments matches
    the type hint or not

    Parameters
    ----------
    function
        Callable object for the function to be checked

    Raises
    ------
    TypeError
          If the input type is not match with the arguments

    """
    @wraps(function)
    def decorated(*args) -> Any:
        for argument_idx, name in enumerate(function.__code__.co_varnames):

            argtype = function.__annotations__.get(name)

            if isinstance(argtype, type):
                if argument_idx < len(args):
                    if not isinstance(args[argument_idx], argtype):
                        raise TypeError(f'{name} should be a {argtype} not '
                                        '{type(args[argument_idx])}')
        return function(*args)

    return decorated
