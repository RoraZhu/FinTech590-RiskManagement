import pandas as pd
import numpy as np
import numpy.linalg


def exponentialWeights(size, lam):
    weight = []
    # cumWeight = []
    totalWeight = 0
    for i in range(size):
        weight.append((1-lam)*(lam**(i+1)))
        totalWeight += weight[i]
        # cumWeight.append(totalWeight)
    for i in range(size):
        weight[i] = weight[i]/totalWeight
        # cumWeight[i] = cumWeight[i]/totalWeight
    return weight


def expCovForPair(weight, x, y):
    x = list(x)
    y = list(y)
    xMean = np.mean(x)
    yMean = np.mean(y)
    cov = 0
    for i in range(len(weight)):
        cov += weight[len(weight)-i-1]*(x[i] - xMean) * (y[i] - yMean)
    return cov


def expCovForFrame(data, lam):
    weight = exponentialWeights(len(data), lam)
    covMatrix = pd.DataFrame(np.zeros((len(data.columns), len(data.columns))))
    for i in range(len(data.columns)):
        x = data.iloc[:, i]
        covMatrix.iloc[i, i] = expCovForPair(weight, x, x)
        for j in range(i+1, len(data.columns)):
            y = data.iloc[:, j]
            covMatrix.iloc[i, j] = expCovForPair(weight, x, y)
            covMatrix.iloc[j, i] = covMatrix.iloc[i, j]
    return covMatrix


def pcaExplained(covMatrix):
    eigenvalue, eigenvector = numpy.linalg.eigh(covMatrix)
    eigenSum = np.sum(eigenvalue)
    totalEigen = 0
    cumEigen = []
    for i in range(len(eigenvalue)-1, -1, -1):
        if eigenvalue[i] < 0:
            break
        totalEigen += eigenvalue[i]
        cumEigen.append(totalEigen/eigenSum)
    return cumEigen
