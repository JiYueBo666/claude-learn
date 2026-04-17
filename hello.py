#!/usr/bin/env python3
"""
A simple hello module that provides a greeting function.

This module contains a function to generate personalized greetings
for users. It demonstrates proper Python documentation practices
and type hinting conventions.
"""


def hello(name: str) -> None:
    """
    Print a personalized greeting message.

    Args:
        name (str): The name of the person to greet

    Returns:
        None: This function prints directly to stdout

    Example:
        >>> hello("Alice")
        hello, Alice
    """
    print(f"hello, {name}")


def main() -> None:
    """Main function that demonstrates the hello function."""
    hello("World")


if __name__ == "__main__":
    main()