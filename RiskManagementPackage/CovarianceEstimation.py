from math import isclose
import numpy as np


def corrToCov(corr, std):
    std = np.diag(std)
    return std @ corr @ std


def checkCov(cov, **kwargs):
    """
    checkCov returns true if the matrix is a covariance matrix.
    :param kwargs: (see isclose document)
    rel_tol      eg. rel_tol=1e-8
    abs_tol      eg. abs_tol=1e-8
    :return: boolean
    """
    count = len([i for i in range(len(cov)) if isclose(cov.iloc[i, i], 1, **kwargs)])
    if count is len(cov):
        # the input matrix is correlation
        return False
    return True


def covToCorr(cov):
    """
    covToCorr transforms a covariance matrix to a correlation matrix.

    :param cov: 2D array-like, symmetric covariance matrix
    :return: 2D array-like, the correlation matrix
    """
    invSD = np.diag(1 / np.sqrt(np.diag(cov)))
    return invSD @ cov @ invSD


def exponentialWeights(size, lam):
    weights = np.array([])
    for i in range(size):
        weights = np.append(weights, (1 - lam) * (lam ** (i + 1)))
    weights /= sum(weights)
    return weights


def getEWCov(x, y, lam):
    weights = exponentialWeights(len(x), lam)
    return ((x - x.mean()) * weights * (y - y.mean())).sum()


def getEWCovMatrix(matrix, lam):
    mat = matrix - np.vstack(matrix.mean(axis=1))
    weights = exponentialWeights(mat.shape[1], lam)
    return mat @ np.diag(weights) @ mat.T


def ewCovToStd(matrix, lam):
    exCov = getEWCovMatrix(matrix, lam)
    return np.sqrt(np.diag(exCov))


def ewCovToCorr(matrix, lam):
    exCov = getEWCovMatrix(matrix, lam)
    invSD = np.diag(1 / np.sqrt(np.diag(exCov)))
    return invSD @ exCov @ invSD
