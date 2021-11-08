import unittest
import numpy as np
from qr import *

class TestQR(unittest.TestCase):
    total_grade_ = 0
    total_grade_file = open("total_grade.txt","w")
    
    @classmethod
    def setUpClass(cls):
        TestQR.total_grade_file.write(str(0))
        
    @classmethod
    def tearDownClass(cls):
        TestQR.total_grade_file.truncate(0)
        print('\nTotal: ', TestQR.total_grade_)
        TestQR.total_grade_file.write(str(TestQR.total_grade_))
        TestQR.total_grade_file.close()
        
    def IncGrade(p):
        TestQR.total_grade_ += p
    
    def test_qr_ident(self):
        identity = np.eye(16)
        Q, R = qr(identity)
        self.assertTrue(np.allclose(Q, identity))
        TestQR.IncGrade(5)
        self.assertTrue(np.allclose(R, identity))
        TestQR.IncGrade(5)

    def test_qr_random(self):
        A = np.random.random((6,4))
        Q, R = qr(A)
        self.assertTrue(Q.shape == (6, 4))
        TestQR.IncGrade(4)
        self.assertTrue(R.shape == (4, 4))
        TestQR.IncGrade(4)
        self.assertTrue(np.allclose(Q @ R, A))
        TestQR.IncGrade(4)
        self.assertTrue(np.allclose(Q.T @ Q, np.eye(4)))
        TestQR.IncGrade(4)
        self.assertTrue(np.allclose(R, np.triu(R)))
        TestQR.IncGrade(4)

    def test_qr_full_random(self):
        A = np.random.random((6,4))
        Q, R = qr_full(A)
        self.assertTrue(Q.shape == (6, 6))
        TestQR.IncGrade(4)
        self.assertTrue(R.shape == (6, 4))
        TestQR.IncGrade(4)
        self.assertTrue(np.allclose(Q @ R, A))
        TestQR.IncGrade(4)
        self.assertTrue(np.allclose(Q.T @ Q, np.eye(6)))
        TestQR.IncGrade(4)
        self.assertTrue(np.allclose(R, np.triu(R)))
        TestQR.IncGrade(4)

    def test_backsub_random(self):
        R = np.triu(np.random.random((6, 6)))
        x = np.random.random((6, 1))
        b = R @ x
        x_hat = back_substitute(R, b)
        self.assertTrue(np.allclose(x, x_hat))
        TestQR.IncGrade(10)

    def test_solve_random(self):
        A = np.random.random((6, 6))
        x = np.random.random((6, 1))
        b = A @ x
        x_hat = solve(A, b)
        self.assertTrue(np.allclose(x, x_hat))
        TestQR.IncGrade(20)

    def test_solve_lstsq(self):
        A = np.random.random((8, 4))
        b = np.random.random((8, 1))
        x = np.linalg.lstsq(A, b)[0]
        x_hat = lstsq(A, b)
        self.assertTrue(np.allclose(x, x_hat))
        TestQR.IncGrade(20)

if __name__=='__main__':
    unittest.main()
