import math
import random
from typing import Callable, Iterable


def sigmoid(x: float) -> (float, float):
    result = None
    prime = None
    if x >= 0:
        result = (1. / (1. + math.exp(-x)))
    else:
        result = (math.exp(x) / (1. + math.exp(x)))

    prime = result * (1.0 - result)
    return result, prime


def relu(x: float) -> (float, float):
    if x > 0:
        return x, 1
    else:
        return 0, 0


def matrix(rows: int, cols: int, fill: float | Callable[[], float] = 0):
    matrix = list()
    for i in range(rows):
        if callable(fill):
            matrix.append([fill() for _ in range(cols)])
        else:
            matrix.append([fill for _ in range(cols)])
    return matrix


def matMul(left, right):
    result = list()
    for i in right:
        total = 0
        for j in left:
            total += left[j] * right[i][j]


def vectorMul(left: list, right: list | float) -> list[float]:
    if type(right) is list:
        if len(left) != len(right):
            raise ValueError("The two vectors need to be the same length.")
        return [left[i] * right[i] for i in range(len(left))]
    else:
        return [left[i] * right for i in range(len(left))]


def vectorDiv(left: list, right: list | float) -> list[float]:
    if type(right) is list:
        if len(left) != len(right):
            raise ValueError("The two vectors need to be the same length.")
        return [left[i] / right[i] for i in range(len(left))]
    else:
        return [left[i] / right for i in range(len(left))]


def vectorDot(left: list, right: list) -> float:
    vector = vectorMul(left, right)
    return sum(vector)


def vectorAdd(left: list, right: list) -> list[float]:
    if type(right) is list:
        if len(left) != len(right):
            raise ValueError("The two vectors need to be the same length.")
        return [left[i] + right[i] for i in range(len(left))]
    else:
        return [left[i] + right for i in range(len(left))]


def vectorSub(left: list, right: list | float) -> list[float]:
    if type(right) is list:
        if len(left) != len(right):
            raise ValueError("The two vectors need to be the same length.")
        return [left[i] - right[i] for i in range(len(left))]
    else:
        return [left[i] - right for i in range(len(left))]


def argmax(x: list[float] | list[int]):
    highIndex = 0
    high = x[0]

    for i in range(len(x)):
        if x[i] > high:
            high = x[i]
            highIndex = i

    return highIndex
