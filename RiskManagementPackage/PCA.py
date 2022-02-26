import pandas as pd
import numpy as np


class PCA:
    def __init__(self, cov):
        self.cov = cov
        self.eigenvalue, self.eigenvector = np.linalg.eigh(cov)

        desc_ranking = np.argsort(self.eigenvalue)[::-1]
        self.eigenvalue = self.eigenvalue[desc_ranking]
        self.eigenvector = self.eigenvector[:, desc_ranking]

    def getEigenvalue(self):
        return self.eigenvalue

    def getEigenvector(self):
        return self.eigenvector

    def findPCARoot(self, explainPower):
        eigenSum = np.sum(self.eigenvalue)
        totalEigen = 0
        i = 1
        while i <= len(self.cov):
            totalEigen += self.eigenvalue[len(self.eigenvalue) - i] / eigenSum
            if totalEigen >= (explainPower - 1e-8):
                break
            i += 1
        eigenvalue = self.eigenvalue[(len(self.eigenvalue) - i):]
        eigenvalue = eigenvalue[::-1]
        eigenvector = (self.eigenvector.T[(len(self.eigenvector) - i):len(self.eigenvector)]).T
        eigenvector = (eigenvector.T[::-1]).T
        return eigenvector @ np.diag(np.sqrt(eigenvalue))

    def pcaExplained(self):
        eigenSum = np.sum(self.eigenvalue)
        eigenvalueReverse = self.eigenvalue[::-1]
        return (eigenvalueReverse / eigenSum).cumsum()