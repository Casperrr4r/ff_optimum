# -*- coding: utf-8 -*-
import atexit
import os
import subprocess
from threading import Condition
import time
from typing import Optional

from ipyparallel import Client
from ipyparallel.error import (CompositeError, NoEnginesRegistered,
                               TimeoutError)

from ff_optimum.cores.utilities import argument_type_check, EventLogger
from ff_optimum.cores.utilities.file_path import (file_get_home_directory,
                                                  file_is_directory_valid,
                                                  file_set_path)

__all__ = ['start_client', 'wait_until_client_ready',
           'get_client', 'shutdown_client', 'is_client_ready']

logger = EventLogger(__name__)


class SingletonClient(object):

    """
    Class implementing shared resources of ipyparallel Client

    Attributes
    ----------
    instance: Client
        the singleton instance of ipyparallel client

    main_pid: int
        the main process pid

    cluster_pid: int
        the pid of ipcluster

    is_ready: boolean

    nproc: int
        number of processors

    profile: str
        name of profile

    Methods
    -------
    get_instance
        get the singleton of the Client

    is_ready
        check whether the client is ready to use

    shutdown_singleton_client
        close the ipyparallel

    """

    __instance = None

    __main_pid = None

    __cluster_pid = None

    __is_ready = False

    __nproc = int()

    __profile = str()

    __use_count = int()

    def __new__(cls, *args, **kwargs) -> object:

        obj = super(SingletonClient, cls).__new__(cls)

        obj.__slots__ = ['__instance', '__main_pid', '__cluster_pid',
                         ' __is_ready', '__nproc', '__profile', '__use_count']

        return obj

    def __init__(self, number_of_processors: Optional[int]=None,
                 profile_name: Optional[str]=None) -> None:

        if (SingletonClient.__instance is None and
                number_of_processors is not None):

            try:

                SingletonClient.__main_pid = os.getpid()

                SingletonClient.__create_profile(profile_name)

                SingletonClient.__profile = profile_name

                SingletonClient.__nproc = number_of_processors

                SingletonClient.__cluster_pid = \
                    self.__start_cluster(number_of_processors, profile_name)

                SingletonClient.__instance = self.__prepare_client()

                SingletonClient.__is_ready = True

            except (CompositeError, ModuleNotFoundError, StopIteration) as e:

                logger.error(e)

                logger.info('Shutting down parallel engine')

                SingletonClient.shutdown_engine()

                raise

    @staticmethod
    @argument_type_check
    def __create_profile(profile_name: str) -> None:
        """
        Check whether profile is exists or not
        Create profile if it is not exists

        Parameters
        ----------
        profile_name
            string of profile

        Returns
        -------
        None

        See Also
        --------
        cores.utilities.argument_type_check.argument_type_check
        cores.utilities.file_path.file_set_path
        cores.utilities.file_path.file_is_directory_valid
        """

        try:

            profile_directory = file_set_path(file_get_home_directory(),
                                              f'.ipython/profile_'
                                              f'{profile_name}')

            file_is_directory_valid(profile_directory)

        except (NotADirectoryError, FileNotFoundError) as e:

            logger.info(f'Profile {profile_name} is not found, '
                        f'start creating profile')

            arguments = ['ipython', 'profile', 'create', '--parallel',
                         f'--profile={profile_name}']

            profile_proc = subprocess.Popen(
                arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    @staticmethod
    def __start_cluster(
            number_of_processors: int,
            profile_name: str) -> int:
        """
        Start ipyparallel by creating sub proess

        Arguments
        ---------
        number_of_processors
            number of processors to be used

        profile_name
            name of the profile

        """

        argument = ['ipcluster', 'start', f'--n={number_of_processors}']

        cluster_proc = subprocess.Popen(
            argument, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        logger.info(f'Starting {number_of_processors} engine, '
                    f'pid:{cluster_proc.pid}')

        return cluster_proc.pid

    @staticmethod
    def __prepare_client(max_trial: int=5, interval: int=10) -> Client:
        """
        Spin and ping the client after sleeping for the interval

        Arguments
        ---------
        max_trial
            Number of maxium trial

        interval
            Time until next ping in seconds

        Raises
        ------
        StopIteration
            When max trial is achieved
        """

        trial = 0

        while trial < max_trial:

            try:

                client = Client()

                time.sleep(interval)

                return client

            except (NoEnginesRegistered, TimeoutError) as e:
                trial += 1

        raise StopIteration(
            'Connecting engine to controller exceed maximum time of trial')

    @staticmethod
    def get_instance() -> Client:
        """
        Get the singleton instance

        Arguments
        ---------
        None

        Returns
        -------
        Client
              The singleton client
        """

        SingletonClient.__instance.close()

        SingletonClient.__instance = Client()

        return SingletonClient.__instance

    @staticmethod
    def is_ready() -> bool:
        """
        Arguments
        ---------
        None

        Returns
        -------
        True if the Client is initated
        Otherwise False
        """

        return SingletonClient.__is_ready

    @staticmethod
    def shutdown_singleton_client() -> None:
        """
        Create a subprocess to shutdown the client

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        is_main_proc = (SingletonClient.__main_pid == os.getpid())

        if SingletonClient.__is_ready and is_main_proc:

            argument = ['ipcluster', 'stop']

            shutdown_proc = subprocess.Popen(
                argument, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            logger.info('Parallel engine is shutting down')


@argument_type_check
def start_client(condition_variable: Condition, nproc: int,
                 profile: str) -> None:
    """
    Init the SingletonClient, notify all the waiting thread after the client
    is initated

    Parameters
    ----------
    condition_variable
        condition object for notifying waiting thread

    nproc
          number of processors

    profile
          name of the profile

    Returns
    -------
    None

    See Also
    --------
    cores.utilities.argument_type_check.argument_type_check
    """

    with condition_variable:

        SingletonClient(nproc, profile)

        condition_variable.notifyAll()

        logger.info('Parallel engine initialize finished')


@argument_type_check
def wait_until_client_ready(condition_variable: Condition,
                            timeout: int=90) -> None:
    """
    Wait until the Client is ready with a timeout 90 seconds

    Parameters
    ----------
    condition_variable
        condition object for notifying waiting thread

    Returns
    -------
    None

    See Also
    --------
    cores.utilities.argument_type_check.argument_type_check
    """

    with condition_variable:

        condition_variable.wait(timeout)

        logger.info('Parallel engine is now ready')


def is_client_ready() -> bool:
    """
    Arguments
    ---------
    None

    Returns
    -------
    True if the ipyparallel client is ready otherwise False

    See Also
    --------
    SingletonClient().is_ready()
    """

    return SingletonClient().is_ready()


def get_client() -> object:
    """
    Shutdown the client

    Arguments
    ---------
    None

    Returns
    -------
    Client
        ipyparallel client object

    See Also
    --------
    SingletonClient().get_instance()
    """

    return SingletonClient().get_instance()


@atexit.register
def shutdown_client() -> None:
    """
    Shutdown the client when the program terminate

    Arguments
    ---------
    None

    Returns
    -------
    None

    See Also
    --------
    SingletonClient().shutdown_singleton_client()
    """

    SingletonClient().shutdown_singleton_client()
