# -*- coding: utf-8 -*-
from typing import Any

__all__ = ['CommandsHolder']


class CommandsHolder(object):

    """
    Class storing and excuting the compiled commands

    Attributes
    ----------
    compiled_commands: dict
        Dictionary object storing the compiled commands
    excutor: callable
        The callanle for executing the compiled commands

    Methods
    -------
    execute_commands(inputs)
        execute the compiled commands

    """

    __slots__ = ['__compiled_commands', '__excutor', '__temp_directory']

    def __init__(self) -> None:

        self.__compiled_commands = None

        self.__excutor = None

        self.__temp_directory = None

    def __enter__(self) -> object:
        return self

    def __exit__(self, exc_ty, exc_val, tb) -> None:
        pass

    @property
    def compiled_commands(self) -> Any:
        """
        The compiled commands

        getter:
            retrun the compiled_commands stored in the instance
        setter:
            set the input compiled_commands to the one stored in the
            instance
        """
        return self.__compiled_commands

    @compiled_commands.setter
    def compiled_commands(self, compiled_commands: Any) -> None:
        self.__compiled_commands = compiled_commands

    @property
    def excutor(self) -> callable:
        """
        The executor function

        getter:
            retrun executor function stored in the instance
        setter:
            set executor function to the one stored in the
            instance
        """
        return self.__excutor

    @excutor.setter
    def excutor(self, func: callable) -> None:
        self.__excutor = func

    @property
    def temp_directory(self) -> str:
        return self.__temp_directory

    @temp_directory.setter
    def temp_directory(self, temp_directory) -> None:
        self.__temp_directory = temp_directory

    def execute_commands(self, inputs: Any) -> Any:
        """
        The execute the executor function stored in the instance

        Parameters
        ----------
        inputs
            extra inputs for the executor function

        Returns
        -------
        the return values of the executor function

        """
        return self.__excutor(self.compiled_commands, inputs,
                              self.__temp_directory)
