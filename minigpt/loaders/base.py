"""Base load to handle text data"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TextDataBase(ABC):
    """class for keeping text data"""

    filename: str
    verbose: bool = False

    def __init__(self, filename, verbose=False):
        self.filename = filename
        self.verbose = verbose

    @abstractmethod
    def load_data(self):
        """Load Data from file"""

    @abstractmethod
    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""

    @abstractmethod
    def decode(self, l) -> str:
        """decode a list of integers back to a string"""

    @classmethod
    def get_loader(cls, type, verbose=False) -> TextDataBase | None:
        if type == "s_char":
            from .shakespeare_char import CharDataTinyShakespeare

            return CharDataTinyShakespeare(verbose=verbose)
        elif type == "s_word":
            from .shakespeare_subword import TextDataTinyShakespeare

            return TextDataTinyShakespeare(verbose=verbose)
        else:
            print(f"Unknown TextData Type: {type}")
            raise ValueError(f"Unknown TextData Type: {type} - Should be `s_char` or `s_word`")
