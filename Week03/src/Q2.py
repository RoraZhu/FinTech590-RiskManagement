import numpy as np
import pandas as pd
import numpy.linalg
from math import isclose


def CholeskyPSD(A):
    A = pd.DataFrame(A)
    L = pd.DataFrame(np.zeros((len(A), len(A))))
    for j in range(len(A)):
        diagMinus = 0
        if j > 0:
            diagMinus = (L.iloc[j, 0:j] * L.iloc[j, 0:j]).sum()
        tempDiag = A.iloc[j, j] - diagMinus
        if -1e-8 <= tempDiag <= 0:
            tempDiag = 0
        L.iloc[j, j] = np.sqrt(tempDiag)
        if -1e-8 <= tempDiag <= 1e-8:
            L.iloc[j, (j + 1):] = 0
            continue
        else:
            for i in range(j + 1, len(A)):
                offDiagMinus = 0
                if j > 0:
                    offDiagMinus = (L.iloc[i, 0:j] * L.iloc[j, 0:j]).sum()
                L.iloc[i, j] = (A.iloc[i, j] - offDiagMinus) / L.iloc[j, j]
    return L


def covToCorr(A):
    count = 0
    for i in range(len(A)):
        if isclose(A.iloc[i, i], 1, abs_tol=1e-8):
            count = count + 1
    if count is not len(A):
        invSD = np.diag(1 / np.sqrt(np.diag(A)))
        return invSD, np.matmul(np.matmul(invSD, A), invSD)
    else:
        return None, A


def nearPSD(A):
    invSD, A = covToCorr(A)
    eigenvalue, eigenvector = numpy.linalg.eigh(A)
    for i in range(len(eigenvalue)):
        if eigenvalue[i] < 0:
            eigenvalue[i] = 0
    scalingMatrix = 1 / np.matmul((eigenvector * eigenvector), eigenvalue)
    scalingMatrix = np.diag(scalingMatrix)

    eigenvalueMatrix = np.diag(eigenvalue)

    factor = np.matmul(np.matmul(np.sqrt(scalingMatrix), eigenvector), np.sqrt(eigenvalueMatrix))
    nearCorr = np.matmul(factor, np.transpose(factor))

    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        nearCorr = np.matmul(np.matmul(invSD, nearCorr), invSD)
    return nearCorr


def firstProjection(A):
    for i in range(len(A)):
        A[i][i] = 1
    return A


def secondProjection(A):
    eigenvalue, eigenvector = numpy.linalg.eigh(A)
    eigenvalue[eigenvalue < 0] = 0
    eigenvalueMatrix = np.diag(eigenvalue)
    return np.matmul(np.matmul(eigenvector, eigenvalueMatrix), np.transpose(eigenvector))


def frobeniusNorm(A):
    return (A * A).sum().sum()


def highamNearPSD(A, maxIterations, tolerance):
    gamma = float("inf")
    Y = A
    deltaS = pd.DataFrame(np.zeros((len(A), len(A))))
    for k in range(maxIterations):
        R = Y - deltaS
        X = secondProjection(R)
        deltaS = X - R
        Y = firstProjection(X)
        tempGamma = frobeniusNorm(Y - A)
        if np.abs(tempGamma - gamma) < tolerance:
            print("Convergence succeeds.")
            break
        gamma = tempGamma
    return pd.DataFrame(Y)

# A = [[1, 1, 0.9, 0.9, 0.9],
#      [1, 1, 0.9, 0.9, 0.9],
#      [0.9, 0.9, 1, 0.9, 0.9],
#      [0.9, 0.9, 0.9, 1, 0.9],
#      [0.9, 0.9, 0.9, 0.9, 1]]
#
# A2 = [[1.17399,   -0.629631,   -0.278932,   -0.081448,  -0.73514],
#       [-0.629631,   1.3182,      0.0180896,   0.446047,   0.139309],
#       [-0.278932,   0.0180896,   0.918102,    0.360836,   0.258613],
#       [-0.081448,   0.446047,    0.360836,    0.894764,  -0.23519],
#       [-0.73514,    0.139309,    0.258613,   -0.23519,    0.522607]]
# A2 = pd.DataFrame(A2)
# PSD = nearPSD(A2)
# print(PSD)
# print(CholeskyPSD(PSD))

