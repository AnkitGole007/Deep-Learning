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
    return np.inner(x,y)

def problem_1e (A, i):
    return np.sum(A[i,::2])

def problem_1f (A, c, d):
    return np.mean(A[np.nonzero((A>=c) & (A<=d))])

def problem_1g (A, k):
    eigenvalues, eigenvectors = LA.eig(A)
    return eigenvectors[:,np.argsort(-np.abs(eigenvalues))[:k]]

def problem_1h (x, k, m, s):
    return np.random.multivariate_normal((x+m*np.ones((len(x)))).flatten(),(s*np.identity(len(x))),k).T

def problem_1i (A):
    return A[:,np.random.shuffle(A.transpose())]

def problem_1j (x):
    return np.std(x)

def problem_1k (x, k):
    return np.repeat(x,k,axis=0)

def problem_1l (X, Y):
    return ...

# A1 = np.array([[1,2],[3,2]])
# A2 = np.array([[2,1],[2,4]])
# A3 = np.array([[4,1],[3,7]])
A4 = np.array([[1,2,3,4],[2,3,1,5],[1,1,4,2],[3,1,7,2]])
#A5 = np.array([[1],[2],[3],[4]])

A = np.array([[1, 3], [5, 7]])
B = np.array([[4, 5], [6, 9]])
C = np.array([[2, 5], [4, 8]])
D = np.array([[1,3,4,12,15,34], [4, 2, 53,23,65, 83], [12,23,34,45,56,67]])

print("1a.",problem_1a(A,B))
print("\n 1b.",problem_1b(A,B,C))
print("\n 1c.",problem_1c(A,B,C))
print("\n 1d.",problem_1d(A,B))
print("\n 1e.",problem_1e(D,1))
print("\n 1f.",problem_1f(D,2,15))

arr = [[5,-10,-5],[2,14,2],[-4,-8,6]]
arr1 = np.diag((1, 2, 3))
print("\n 1g.",problem_1g(arr,1))
print("\n 1g.",problem_1g(arr1,2))

x = np.array([1,2,3,4,5]).T
m = 2
s = 3
k = 4
print("\n 1h.",problem_1h(x,4,2,3))
print("\n 1i.",problem_1i(A4))
# print("\n 1j.",problem_1j(A5))
print("\n 1k.",problem_1k(D[0],5))