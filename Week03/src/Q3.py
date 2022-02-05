import numpy as np
import numpy.linalg


def generateStd(cov):
    return np.sqrt(np.diag(cov))


def generateCorr(cov):
    invSD = np.diag(1 / np.sqrt(np.diag(cov)))
    return invSD @ cov @ invSD


def generateCov(std, corr):
    std = np.diag(std)
    return std @ corr @ std


def findPCARoot(cov, explainPower):
    # cov must be pd or psd
    eigenvalue, eigenvector = numpy.linalg.eigh(cov)
    eigenSum = np.sum(eigenvalue)
    totalEigen = 0
    i = 1
    while i <= len(cov):
        totalEigen += eigenvalue[len(eigenvalue) - i] / eigenSum
        if totalEigen >= (explainPower - 1e-8):
            break
        i += 1
    eigenvalue = eigenvalue[(len(eigenvalue) - i):]
    eigenvalue = eigenvalue[::-1]
    eigenvector = (eigenvector.T[(len(eigenvector) - i):len(eigenvector)]).T
    eigenvector = (eigenvector.T[::-1]).T
    return eigenvector @ np.diag(np.sqrt(eigenvalue))



def simulateMultiNormal(rows, columns):
    random = np.random.normal(size=rows*columns)
    return random.reshape(rows, columns)


def simulateCov(root, numOfDraws):
    columns = np.shape(root)[1]
    stdNormal = simulateMultiNormal(columns, numOfDraws)
    simulatedData = root @ stdNormal
    newCov = np.cov(simulatedData)
    return newCov