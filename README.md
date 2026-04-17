# My Python Package

A sample Python package demonstrating proper package structure with utility functions and tests.

## Package Structure

```
.
├── __init__.py          # Package initialization
├── utils.py             # Utility functions implementation
├── tests/
│   └── test_utils.py    # Unit tests for utility functions
└── README.md            # This file
```

## Installation

You can install this package by cloning the repository and installing in development mode:

```bash
git clone <repository-url>
cd my-python-package
pip install -e .
```

## Usage

```python
from my_package import add_numbers, multiply_numbers, reverse_string, is_palindrome, get_even_numbers

# Basic arithmetic
result1 = add_numbers(5, 3)        # Returns 8
result2 = multiply_numbers(4, 7)   # Returns 28

# String operations
reversed = reverse_string("hello")  # Returns "olleh"
is_pal = is_palindrome("radar")     # Returns True

# List operations
evens = get_even_numbers([1, 2, 3, 4, 5, 6])  # Returns [2, 4, 6]
```

## Available Functions

### add_numbers(a, b)
- Adds two numbers together
- Returns: Sum of a and b

### multiply_numbers(a, b)
- Multiplies two numbers together  
- Returns: Product of a and b

### reverse_string(s)
- Reverses a string
- Returns: Reversed string

### is_palindrome(s)
- Checks if a string is a palindrome (ignores spaces and case)
- Returns: Boolean indicating if string is palindrome

### get_even_numbers(numbers)
- Filters even numbers from a list
- Returns: List containing only even numbers

## Running Tests

To run the test suite:

```bash
python -m unittest discover tests/
```

Or run a specific test file:

```bash
python -m unittest tests/test_utils.py
```

## Development

This package follows standard Python packaging conventions:
- Package code is in the root directory
- Tests are in the `tests/` directory
- `__init__.py` makes the directory a Python package
- All functions include docstrings with examples
- Comprehensive unit tests with edge cases