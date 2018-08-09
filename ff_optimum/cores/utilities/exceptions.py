# -*- coding: utf-8 -*-
__all__ = ['FileEmptyError', 'XmlNodeNotFoundError']


class FileEmptyError(Exception):

    """ raise when a file is empty """
    pass


class XmlNodeNotFoundError(Exception):

    """ raise when xml node is not found """
    pass
