# import the necessary packages
import os


class config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
