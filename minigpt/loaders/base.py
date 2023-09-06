"""Base load to handle text data"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TextDataBase(ABC):
    """class for keeping text data"""

    filename: str

    def __init__(self, filename):
        self.filename = filename

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
    def get_loader(cls, type) -> TextDataBase | None:
        if type == "shakespeare_char":
            from .shakespeare_char import TextDataTinyShakespeare

            return TextDataTinyShakespeare()
        else:
            print("Unknown Data Type: {type}")
            return None
