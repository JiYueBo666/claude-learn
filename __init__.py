"""
My Python Package

A simple Python package with utility functions.
"""

# Import key functions for easier access
from .utils import (
    add_numbers,
    multiply_numbers,
    factorial,
    is_even,
    reverse_string
)

# Package version
__version__ = "1.0.0"
__all__ = ["add_numbers", "multiply_numbers", "factorial", "is_even", "reverse_string"]