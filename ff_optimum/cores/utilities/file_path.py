# -*- coding: utf-8 -*-
import json
import os
import os.path
from pathlib import Path
import platform
import stat
from typing import Optional, Union

from ff_optimum.cores.utilities.argument_type_check import argument_type_check
from ff_optimum.cores.utilities.exceptions import FileEmptyError

__all__ = ['file_create_directory', 'file_get_filenames_from_directory',
           'file_get_filename_from_path', 'file_get_home_directory',
           'file_is_directory_valid', 'file_is_file_path_valid',
           'file_read_json', 'file_remove_file', 'file_set_path']


@argument_type_check
def file_create_directory(directory: Union[str, Path]) -> None:
    """
    Create the directory recursivly if the directory is not exist

    Parameters
    ----------
    directory

    Returns
    -------
    None

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    """

    if not __file_is_directory_exists(Path(directory)):
        os.makedirs(directory)


@argument_type_check
def file_get_filenames_from_directory(directory: str,
                                      extension: str='') -> list:
    """
    Search through the directory and find the file with the file extension

    Parameters
    ----------
    directory
        file directory to be searched

    extension
        file extension to be searched

    Returns
    -------
    list constaining all the files with the extension in the directory

    See Also
    --------
    """

    file_is_directory_valid(directory)

    paths = []

    for root, directory, file_names in os.walk(directory):
        paths.extend(list(map(lambda name: file_set_path(root, name),
                     filter(lambda file_name:
                            file_name.endswith(extension), file_names))))

    return paths


@argument_type_check
def file_get_filename_from_path(filepath: str) -> str:
    """
    Get the filename from a file path

    Parameters
    ----------
    filepath

    Returns
    -------
    filename

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    """

    _, filename = os.path.split(filepath)

    return filename


def file_get_home_directory() -> str:
    """
    Get the home directory of the current user

    Parameters
    ----------
    None

    Returns
    -------
    The home directory of the current user
    """
    return os.path.expanduser('~')


@argument_type_check
def file_is_directory_valid(directory: str) -> bool:
    """
    Check whether the input directory is valid or not

    Parameters
    ----------
    directory
          string containing the directory

    Returns
    -------
    True
          If the directory is valid

    Raises
    ------
    NotADirectoryError
          If the input directory is not a directory

    FileNotFoundError
          If the input directory is not exists

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    """

    path = Path(directory)

    if not path.is_dir():

        if './ipython' in directory:
            if bool(os.stat(directory).st_file_attribute &
                    stat.FILE_ATTRIBUTE_HIDDEN):
                return True

        raise NotADirectoryError(f"{path} is not a directory")

    if not __file_is_directory_exists(path):
        raise FileNotFoundError(f"{path} is not exist")

    return True


@argument_type_check
def file_is_file_path_valid(file_path: str,
                            extension: Optional[str]=None) -> bool:
    """
    Check whether the file_path is valid or not

    Parameters
    ----------
    file_path
          a string containing the file path

    extension
         a string containing the file extension

    Returns
    -------
    True
          If the directory is valid

    Raises
    ------
    FileNotFoundError
          If the input directory is not exists

    ValueError
         If the extension of the file path and the input extension
         is not match

    FileEmptyError
         If the file is an empty file

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    utilities.file_system.file_path.__file_is_directory_exists
    utilities.file_system.file_path.__file_check_extension
    utilities.file_system.file_path.__file_is_empty
    """
    path = Path(file_path)

    if not __file_is_directory_exists(path):
        raise FileNotFoundError(f"{path} is not exist")

    if extension:
        if not __file_check_extension(file_path, extension):
            raise ValueError(f"{file_path} and extension: {extension} "
                             "are not matched")

    if __file_is_empty(path):
        raise FileEmptyError("{path} is an empty file")

    return True


@argument_type_check
def file_read_json(file_path: str) -> None:
    """
    Load json config file

    Parameters
    ----------
    config_path
        File path of the json config path

    Returns
    -------
    dict
        Dictionary object storing the loaded json config
    """

    file_is_file_path_valid(file_path, 'json')

    with open(file_path, 'r') as fp:
        return json.load(fp)


@argument_type_check
def file_remove_file(file_path: str) -> None:
    """
    Remove the file path is valid or not

    Parameters
    ----------
    file_path
          a string containing the file path

    Returns
    -------
    None

    Raises
    ------
    NotADirectoryError
          If the input file path is not a directory

    FileNotFoundError
          If the input file path is not exists

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    utilities.file_system.file_path.file_is_directory_valid
    """

    path = Path(file_path)

    if file_is_directory_valid(path):
        os.remove(path)


@argument_type_check
def file_set_path(directory: str, filename: str) -> str:
    """
    Join the file path according to the input directory and filename

    Parameters
    ----------
    directory
          a string containing the directory

    filename
          a string containing the filename

    Returns
    -------
    String containing the joined file path

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    """

    if platform.system() == 'windows':
        return f'{directory}\\{filename}'
    else:
        return os.path.join(directory, filename)


def __file_check_extension(file_path: str, extension: str) -> bool:
    """
    Check whether the input file path contains the target extension

    Parameters
    ----------
    file_path
          string containing the input file path

    extension
          string containing the target extension

    Returns
    -------
    bool
          whether the input file path contains the target extension
    """
    return file_path.endswith(extension)


def __file_is_empty(path: Path) -> bool:
    """
    Check whether the input file path is an empty file

    Parameters
    ----------
    path
          Path object containing the input file path

    Returns
    -------
    bool
          whether the input file path is an empty file
    """
    return path.stat().st_size == 0


def __file_is_directory_exists(path: Path) -> bool:
    """
    Check whether the input file path is exists or not

    Parameters
    ----------
    path
          Path object containing the input file path

    Returns
    -------
    bool
          whether the input file path is exists or not
    """
    return path.exists()
