import numpy as np

"""
There are five functions for you to implement:
1. qr
2. qr_full
3. back_substitute
4. solve
5. least_squares
Read the docstring for each function to find out what's required.  qr_test.py contains tests to pass, but feel free to add your own.  You don't turn in qr_test.py.

numpy has built-in QR factorization in its library.
Do not use library QR functions!  If you call numpy.linalg.qr or scipy.linalg.qr, you will get 0 points.  You may use other numpy library functions, but stay close to the algorithm described in the textbook.
"""


def qr(A):
    """This function takes a single input: a matrix A stored as a 2-d numpy array.  Your function can assume that A has independent columns.

    qr(A) should return a pair of matrices (Q, R) such that:
    + If A is N*K, Q is N*K and R is K*K
    + Q is orthornormal
    + R is upper triangular
    + Q * R = A
    """
    return (Q, R)


def qr_full(A):
    """This function takes a single input: a matrix A stored as a 2-d numpy array.  Your function can assume that A has independent columns.

    This function will be identical to qr() except that it will return a full square orthonormal Q.
    qr_full(A) should return a pair (Q, R) such that:
    + If A is N*K, then Q is N*N (square!) and R is N*K
    + Q is orthornormal
    + R is upper triangular
    + Q * R = A

    You may use your qr function in implementing this function.
    """
    return (Q, R)


def back_substitute(R, b):
    """This function takes two inputs:
    + Matrix R stored as a 2-d numpy array.  Your function can assume that R is upper-triangular with nonzero diagonal.
    + Vector b stored as a 1-d numpy vector.

    This function should return a vector x such that R * x = b.

    Don't use a numpy solve function for this.
    Follow the textbook algorithm 11.1 "Back substitution".
    """
    return x


def solve(A, b):
    """This function takes two inputs:
    + Matrix A stored as a 2-d numpy array.  Your function can assume that A has independent columns.
    + Vector b stored as a 1-d numpy vector.

    This function should return a vector x such that A * x = b.

    You MUST use your qr() and back_substitute() functions to solve this linear system.
    Follow the textbook algorithm 11.2 "Solving linear equations via QR factorization".
    """
    return x


def lstsq(A, b):
    """This function takes two inputs:
    + Matrix A stored as a 2-d numpy array.  Your function can assume that A has independent columns.
    + Vector b stored as a 1-d numpy vector.

    This function should return a vector x such that x is the least-squares solution, minimizing ||Ax - b||.

    You MUST use your own functions to solve this problem, not library functions.
    Follow the textbook algorithm 12.1 "Least squares via QR factorization".
    """
    return x
