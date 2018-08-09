# -*- coding: utf-8 -*-
from typing import Optional
import xml.etree.ElementTree as ET

from ff_optimum.cores.utilities.argument_type_check import argument_type_check
from ff_optimum.cores.utilities.exceptions import XmlNodeNotFoundError
from ff_optimum.cores.utilities.file_path import file_is_file_path_valid

__all__ = ['get_root_from_file', 'get_flag_from_node',
           'get_text_from_node', 'get_text_from_parent',
           'get_text_from_iter']


@argument_type_check
def get_root_from_file(xml_path: str) -> ET.Element:
    """
    get the root of an xml file

    Parameters
    ----------
    xml_path
          string containing the xml file path

    Returns
    -------
    the root of the xml file

    See Also
    --------
    cores.utilities.argument_type_check.argument_type_check
    cores.utilities.file_path.file_is_file_path_valid
    """

    file_is_file_path_valid(xml_path, '.xml')

    tree = ET.ElementTree(file=xml_path)

    return tree.getroot()


@argument_type_check
def get_flag_from_node(node: ET.Element, attribute: str) -> bool:
    """
    get the root of an xml file

    Parameters
    ----------
    node
          xml.etree.ElementTree.Element object of the node to be searched

    attribute
          string containing the attribute to be searched

    Returns
    -------
    The flag of the xml node

    Raises
    ------
    XmlNodeNotFoundError
          If the attribute is not exists in the element

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    """

    element = node.find(attribute)

    if element is not None:
        return element.attrib['flag'].lower() == 'true'
    else:
        raise XmlNodeNotFoundError(f'{attribute} is not found in the xml')


@argument_type_check
def get_text_from_node(node: ET.Element, attribute: str) -> Optional[str]:
    """
    get the root of an xml file

    Parameters
    ----------
    node
          xml.etree.ElementTree.Element object of the node to be searched

    attribute
          string containing the attribute to be searched

    Returns
    -------
    The text of the xml node

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    """

    element = node.find(attribute)

    return element.text if element is not None else None


@argument_type_check
def get_text_from_parent(node: ET.Element, attribute: str) -> list:
    """
    get the text from the children of a xml node by searching the attribute

    Parameters
    ----------
    node
          xml.etree.ElementTree.Element object of the node to be searched

    attribute
          string containing the attribute to be searched

    Returns
    -------
    List of text

    Raises
    ------
    XmlNodeNotFoundError
          If the attribute is not exists in the element

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    """

    parent = node.find(attribute)

    if parent is not None:
        return [child.text for child in parent]
    else:
        raise XmlNodeNotFoundError(f'{attribute} is not found in the xml')


@argument_type_check
def get_text_from_iter(node: ET.Element, attribute: str) -> list:
    """
    get the text of the xml node by iterating through the attribute

    Parameters
    ----------
    node
          xml.etree.ElementTree.Element object of the node to be searched

    attribute
          string containing the attribute to be searched

    Returns
    -------
    List of text

    See Also
    --------
    utilities.argument_type_check.argument_type_check
    """
    return [[child.text for child in it] for it in node.iter(attribute)]
