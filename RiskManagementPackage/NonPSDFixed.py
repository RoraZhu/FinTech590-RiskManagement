import pandas as pd
import numpy as np
from RiskManagementPackage.MatrixNorm import frobeniusNorm


def isPSD(matrix, tolerance=1e-8):
    if abs(matrix - matrix.T).sum().sum() > tolerance:
        raise ValueError("This function is for real symmetric matrices!")

    eigenvalue, eigenvector = np.linalg.eigh(matrix)
    return all(eigenvalue > -tolerance)


def nearPSD(corr):
    """
    nearPSD transforms a non-PSD covariance/correlation matrix to a PSD correlation matrix.
    Not guarantee to be the nearest.

    :param corr: 2D array-like, correlation matrix
    :return: 2D array-like, correlation matrix
    """
    eigenvalue, eigenvector = np.linalg.eigh(corr)
    eigenvalue[eigenvalue < 0] = 0
    scalingMatrix = 1 / ((eigenvector * eigenvector) @ eigenvalue)
    scalingMatrix = np.diag(scalingMatrix)
    eigenvalueMatrix = np.diag(eigenvalue)
    factor = np.sqrt(scalingMatrix) @ eigenvector @ np.sqrt(eigenvalueMatrix)
    refactorMatrix = factor @ factor.T
    return refactorMatrix


def _firstProjection(matrix):
    newMatrix = matrix.copy()
    for i in range(len(newMatrix)):
        newMatrix[i][i] = 1
    return newMatrix


def _secondProjection(matrix):
    eigenvalue, eigenvector = np.linalg.eigh(matrix)
    eigenvalue[eigenvalue < 0] = 0
    return eigenvector @ np.diag(eigenvalue) @ eigenvector.T


def nearestPSD(corr, maxIterations=100, tolerance=1e-9):
    gamma = float("inf")
    Y = corr
    deltaS = pd.DataFrame(np.zeros(corr.shape))
    for k in range(maxIterations):
        R = Y - deltaS
        X = _secondProjection(R)
        deltaS = X - R
        Y = _firstProjection(X)
        tempGamma = frobeniusNorm(Y - corr)
        if np.abs(tempGamma - gamma) < tolerance:
            print("Convergence succeeds.")
            break
        gamma = tempGamma
    return Y