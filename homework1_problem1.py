import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def problem_1a (A, B):
    return (A + B)

def problem_1b (A, B, C):
    return (np.dot(A,B) - C)

def problem_1c (A, B, C):
    return (A*B) + np.transpose(C)

def problem_1d (x, y):
    return np.dot(np.transpose(x),y)

def problem_1e (A, i):
    return np.sum(A[i,::2])

def problem_1f (A, c, d):
    return np.mean(A[np.nonzero((A>=c) & (A<=d))])

def problem_1g (A, k):
    eigenvalues, eigenvectors = LA.eig(A)
    return eigenvectors[:,np.argsort(-np.abs(eigenvalues))[:k]]

def problem_1h (x, k, m, s):
    return np.random.multivariate_normal((x+m*np.ones((len(x),1))).flatten(),(s*np.eye(len(x))),k).T

def problem_1i (A):
    return A[:,np.random.shuffle(A.transpose())]

def problem_1j (x):
    return np.std(x)

def problem_1k (x, k):
    return ...

def problem_1l (X, Y):
    return ...

A1 = np.array([[1,2],[3,2]])
A2 = np.array([[2,1],[2,4]])
A3 = np.array([[4,1],[3,7]])
A4 = np.array([[1,2,3,4],[2,3,1,5],[1,1,4,2],[3,1,7,2]])
A5 = np.array([[1],[2],[3],[4]])

a = problem_1a(A1,A2)
b = problem_1b(A1,A2,A3)
c = problem_1c(A1,A2,A3)
d = problem_1d(A1,A2)
e = problem_1e(A4,2)
f = problem_1f(A1,2,5)
g = problem_1g(A4,2)
h = problem_1h(A5,2,2,3)
i = problem_1i(A4)
j = problem_1j(A5)


print(j)