import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    """Return the product of 2 numbers"""
    return a * b


def id(a: float) -> float:
    """Return the input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers"""
    return a + b


def neg(a: float) -> float:
    """Negates a number"""
    return -a


def lt(a: float, b: float) -> float:
    """Checks if one number is less than another"""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal"""
    if a == b:
        return 1.0
    return 0.0


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers"""
    if a >= b:
        return a
    return b


def is_close(a: float, b: float) -> float:
    """Checks if two numbers are close in value"""
    return (a - b < 1e-2) and (b - a < 1e-2)


def sigmoid(a: float) -> float:
    """Calculates the sigmoid function"""
    return 1 / (1 + math.exp(-a))


def relu(a: float) -> float:
    """Applies the ReLU activation function."""
    if a > 0:
        return a
    return 0


def log(a: float) -> float:
    """Calculates the natural logarithm of a number."""
    if a > 0:
        return math.log(a)
    else:
        raise ValueError("The number must be bigger than 0")


def exp(a: float) -> float:
    """Calculates the exponential of a number."""
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal of a number."""
    if a != 0:
        return 1 / a
    else:
        raise ValueError("the number cannot be 0")


def log_back(a: float, b: float) -> float:
    """Computes the derivative of log times a second arg
    deravtive of log(a) is 1/a, times b
    """
    if a > 0:
        return b / a
    else:
        raise ValueError("The first arg must be bigger than 0")


def inv_back(a: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second arg
    deravtive of max(a,0) is 0 if a <0, 1 if a >=0, times b
    """
    return -(1.0 / a**2) * b


def relu_back(a: float, b: float) -> float:
    """relu_back - Computes the derivative of ReLU times a second arg
    derivative of ReLU(a) is 0 if a < 0, 1 if a >= 0, times b
    """
    if a > 0:
        return b
    else:
        return 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(f: Callable[[float], float], i: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable"""
    return [f(x) for x in i]


def zipWith(
    f: Callable[[float, float], float], i1: Iterable[float], i2: Iterable[float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function"""
    return [f(x, y) for x, y in zip(i1, i2)]


def reduce(
    f: Callable[[float, float], float], i: Iterable[float], initial: float
) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    result = initial
    for x in i:
        result = f(result, x)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, ls, 1.0)
