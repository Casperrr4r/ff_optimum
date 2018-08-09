# -*- coding: utf-8 -*-
import inspect
import logging
import logging.handlers
import os
from typing import Union

from ff_optimum.cores.utilities.file_path import file_set_path

__all__ = ['EventLogger']

EVENT_LOG_MAX_BYTES = 10 * 1024**2

EVENT_LOG_SEVERITY = logging.INFO

EVENT_LOG_BACK_UP_COUNT = 10


class EventLogger(logging.Logger):

    """
    Derived class of logging.Logger

    Attributes
    ----------
    __name: str
        name of the logger

    __path
        filepath of the log file

    __max_bytes
        maximum size before file rotation

    __format
        logging format

    Methods
    -------
    debug
    info
    warning
    error
    critical
    """

    __slots__ = ['__name', '__path', '__max_bytes', '__level', '__format']

    def __init__(self, name: str) -> None:
        """
        Initiate the event logger object

        Parameters
        ----------
        name
              name of the logger

        Returns
        -------
        None

        See Also
        --------
        cores.utilities.file_path.file_set_path
        """

        if isinstance(name, str) is False:
            raise TypeError

        super(EventLogger, self).__init__(name, EVENT_LOG_SEVERITY)

        path = file_set_path(os.getcwd(), 'EventLog.log')

        if not os.path.isfile(path):
            with open(path, 'w') as fp:
                fp.write('Date\t   Time\t\tLevel  Pid  Logger  Message\n')

        fmt = ('%(asctime)-15s.%(msecs)03d [%(levelname)s]' +
               ' %(process)d %(name)s %(message)s')

        self.__format = logging.Formatter(
            fmt=fmt,
                datefmt='%m-%d-%Y %H:%M:%S')

        self.__handler = \
            logging.handlers.RotatingFileHandler(
                path, maxBytes=EVENT_LOG_MAX_BYTES,
                backupCount=EVENT_LOG_BACK_UP_COUNT)

        self.__handler.setFormatter(self.__format)

        self.__handler.setLevel(EVENT_LOG_SEVERITY)

        self.addHandler(self.__handler)

        self.propagate = False

    def logger_decorator(function: callable) -> callable:
        """
        Decorator for including the function name and number of line in
        the logging message

        Parameters
        ----------
        function
            function to be decoratored

        Returns
        -------
        wrapper
            the decoratored function

        """

        def wrapper(self, org_msg, *args, **kwargs) -> callable:

            func_name, line_no = self.__get_function_name_and_line_number()

            msg = f'In {func_name} line {line_no} : {org_msg}'

            return function(self, msg, *args, **kwargs)

        return wrapper

    @staticmethod
    def __get_function_name_and_line_number() -> Union[str, int]:
        """
        Helper function for getting the function name and line number from
        the caller function

        Parameters
        ----------
        None

        Returns
        -------
        caller.f_code.co_name
            function name of the caller
        caller.f_lineno
            line number of the caller

        """

        caller = inspect.currentframe().f_back.f_back

        return caller.f_code.co_name, caller.f_lineno

    @logger_decorator
    def debug(self, msg: str, *args: str, **kwargs: str) -> None:
        return super(EventLogger, self).debug(msg, *args, **kwargs)

    @logger_decorator
    def info(self, msg: str, *args: str, **kwargs: str) -> None:
        return super(EventLogger, self).info(msg, *args, **kwargs)

    @logger_decorator
    def warning(self, msg: str, *args: str, **kwargs: str) -> None:
        return super(EventLogger, self).warning(msg, *args, **kwargs)

    @logger_decorator
    def error(self, msg: str, *args: str, **kwargs: str) -> None:
        return super(EventLogger, self).error(msg, *args, **kwargs)

    @logger_decorator
    def critical(self, msg: str, *args: str, **kwargs: str) -> None:
        return super(EventLogger, self).critical(msg, *args, **kwargs)

    def __del__(self) -> None:
        while self.hasHandlers():
            self.handlers.pop()
