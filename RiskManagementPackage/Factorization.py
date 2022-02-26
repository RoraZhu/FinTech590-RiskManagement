import numpy as np
import pandas as pd


def Cholesky(cov):
    """
    Cholesky returns the root of Cholesky factorization

    :param cov: dataframe, symmetric, can be PD or SPD
    :return: dataframe
    """
    root = pd.DataFrame(np.zeros(cov.shape))
    for j in range(len(cov)):
        diagMinus = 0
        if j > 0:
            diagMinus = (root.iloc[j, 0:j] * root.iloc[j, 0:j]).sum()
        tempDiag = cov.iloc[j, j] - diagMinus
        if -1e-8 <= tempDiag <= 0:
            tempDiag = 0
        root.iloc[j, j] = np.sqrt(tempDiag)
        if -1e-8 <= tempDiag <= 1e-8:
            root.iloc[j, (j + 1):] = 0
        else:
            for i in range(j + 1, len(cov)):
                offDiagMinus = 0
                if j > 0:
                    offDiagMinus = (root.iloc[i, 0:j] * root.iloc[j, 0:j]).sum()
                root.iloc[i, j] = (cov.iloc[i, j] - offDiagMinus) / root.iloc[j, j]
    return root