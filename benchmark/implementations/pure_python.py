"""Pure Python implementations for benchmarking."""

import math


def array_sum(arr: list[float]) -> float:
    return sum(arr)


def matrix_dot(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            a_ik = a[i][k]
            for j in range(n):
                result[i][j] += a_ik * b[k][j]
    return result


def elementwise_sqrt(arr: list[float]) -> list[float]:
    return [math.sqrt(x) for x in arr]
