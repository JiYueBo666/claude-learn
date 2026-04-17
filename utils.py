"""Utility functions module.

This module contains various utility functions for mathematical operations
and string manipulations.
"""

def add_numbers(a: float, b: float) -> float:
    """Add two numbers.
    
    Args:
        a (float): First number
        b (float): Second number
        
    Returns:
        float: Sum of a and b
    """
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers.
    
    Args:
        a (float): First number
        b (float): Second number
        
    Returns:
        float: Product of a and b
    """
    return a * b

def factorial(n: int) -> int:
    """Calculate factorial of a number.
    
    Args:
        n (int): Non-negative integer
        
    Returns:
        int: Factorial of n
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result

def is_even(n: int) -> bool:
    """Check if a number is even.
    
    Args:
        n (int): Integer to check
        
    Returns:
        bool: True if n is even, False otherwise
    """
    return n % 2 == 0


def reverse_string(s: str) -> str:
    """Reverse a string.
    
    Args:
        s (str): String to reverse
        
    Returns:
        str: Reversed string
    """
    return s[::-1]

if __name__ == "__main__":
    # Test the functions
    print(f"add_numbers(2, 3) = {add_numbers(2, 3)}")
    print(f"multiply_numbers(4, 5) = {multiply_numbers(4, 5)}")
    print(f"factorial(5) = {factorial(5)}")
    print(f"is_even(4) = {is_even(4)}")
    print(f"reverse_string('hello') = {reverse_string('hello')}")