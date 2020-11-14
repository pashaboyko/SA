__author__ = 'lex'

import numpy as np
import itertools



def conjugate_gradient_method(A, b, eps):
    '''
    Conjugate Gradient Method that solve equation Ax = b with given accuracy
    :param A:matrix A
    :param b:vector b
    :param eps: accuracy
    :return: solution x
    '''
    n = len(A.T) # number column
    xi1 = xi = np.zeros(shape=(n,1), dtype = float)
    vi = ri = b # start condition
    i = 0 #loop for number iteration
    while True:
        try:
            i+= 1
            ai = float(vi.T*ri)/float(vi.T*A*vi) # alpha i
            xi1 = xi+ai*vi # x i+1
            ri1 = ri-ai*A*vi # r i+1
            betai = -float(vi.T*A*ri1)/float(vi.T*A*vi) # beta i
            vi1 = ri1+betai*vi
            if (np.linalg.norm(ri1)<eps) or i > 10 * n:
                break
            else:
                xi,vi,ri = xi1,vi1,ri1
        except Exception:
            print("There is a problem with minimization.")
    return np.matrix(xi1)





def coordinate_descent(A, b, eps, maxIterations = 10000):
    b = np.array(list(itertools.chain(*b.tolist())))
    A = np.array(A)
    N = A.shape[0]
    x = [0 for i in range(N)]
    xprev = [0.0 for i in range(N)]
    for i in range(maxIterations):
        for j in range(N):
            xprev[j] = x[j]
        for j in range(N):
            summ = 0.0
            for k in range(N):
                if (k != j):
                    summ = summ + A[j][k] * x[k]
            x[j] = (b[j] - summ) / A[j][j]
        diff1norm = 0.0
        oldnorm = 0.0
        for j in range(N):
            diff1norm = diff1norm + abs(x[j] - xprev[j])
            oldnorm = oldnorm + abs(xprev[j])
        if oldnorm == 0.0:
            oldnorm = 1.0
        norm = diff1norm / oldnorm
        if (norm < eps) and i != 0:
            k = []
            for i in x:
                k.append([i])
            return np.matrix(k)
    k = []
    for i in x:
        k.append([i])
    return np.matrix(k)


