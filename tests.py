#!/usr/bin/env python3
import unittest
import numpy as np
from answers_qr import *

class TestQR(unittest.TestCase):
    def test_qr_ident(self):
        identity = np.eye(16)
        Q, R = qr(identity)
        self.assertTrue(np.allclose(Q, identity))
        self.assertTrue(np.allclose(R, identity))

    def test_qr_random(self):
        A = np.random.random((6,4))
        Q, R = qr(A)
        self.assertTrue(Q.shape == (6, 4))
        self.assertTrue(R.shape == (4, 4))
        self.assertTrue(np.allclose(Q @ R, A))
        self.assertTrue(np.allclose(Q.T @ Q, np.eye(4)))
        self.assertTrue(np.allclose(R, np.triu(R)))

    def test_qr_full_random(self):
        A = np.random.random((6,4))
        Q, R = qr_full(A)
        self.assertTrue(Q.shape == (6, 6))
        self.assertTrue(R.shape == (6, 4))
        self.assertTrue(np.allclose(Q @ R, A))
        self.assertTrue(np.allclose(Q.T @ Q, np.eye(6)))
        self.assertTrue(np.allclose(R, np.triu(R)))

    def test_backsub_random(self):
        R = np.triu(np.random.random((6, 6)))
        x = np.random.random((6, 1))
        b = R @ x
        x_hat = back_substitute(R, b)
        self.assertTrue(np.allclose(x, x_hat))

    def test_solve_random(self):
        A = np.random.random((6, 6))
        x = np.random.random((6, 1))
        b = A @ x
        x_hat = solve(A, b)
        self.assertTrue(np.allclose(x, x_hat))

    def test_solve_lstsq(self):
        A = np.random.random((8, 4))
        b = np.random.random((8, 1))
        x = np.linalg.lstsq(A, b)[0]
        x_hat = lstsq(A, b)
        self.assertTrue(np.allclose(x, x_hat))

if __name__=='__main__':
    unittest.main()
