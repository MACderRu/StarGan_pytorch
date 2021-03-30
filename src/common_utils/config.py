from __future__ import annotations

import yaml
import typing as tp


class DotDict(dict):
    """
    Dict-like object, that allows to access values using dot notation:
        dct = DotDict({"one": 1, "two": 2})
        assert dct.one == 1
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, data: tp.Dict[tp.Any, tp.Any]) -> None:
        """
        :param tp.Dict[tp.Any, tp.Any] data: data to populate self
        :returns: None
        """
        super().__init__()
        for key, value in data.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = [
                    DotDict(item) if isinstance(item, dict) else item for item in value
                ]

            self[key] = value


class Config(DotDict):
    """
    DotDict child dedicated to be used to store MarioNet config contents
    """
    @classmethod
    def from_file(cls, filepath: str = "config.yaml") -> Config:
        """
        Reads yaml file and populates self with its contents
        :param str filepath: path to config file
        :returns: populated Config instance
        """
        with open(filepath, "r") as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.Loader)

        return cls(config_dict)
