import numpy as np
import pandas as pd
from scipy.stats import norm, t, spearmanr
from statsmodels.tsa.arima_process import ArmaProcess
from RiskManagementPackage.MLE import mleT
from RiskManagementPackage.Factorization import Cholesky


def armaSimulation(arCoefs, maCoefs, nSample):
    """
    armaSimulation simulates nSample number of data from ARMA(arCoefs, maCoefs)

    :param arCoefs: array-like, start with 1
        eg. [1], [1, 0.2], [1, 0.3, -0.4]

    :param maCoefs: array-like, start with 1
        eg. [1], [1, 0.2], [1, 0.3, -0.4]

    :return: array-like
    """
    if arCoefs is None:
        arCoefs = np.array([1.])
    if maCoefs is None:
        maCoefs = np.array([1.])
    process = ArmaProcess(arCoefs, maCoefs)
    return process.generate_sample(nsample=nSample)


def multiNormalSimulation(nRow, nCol, **kwargs):
    """
    multiNormalSimulation simulates a matrix of random numbers following Normal(loc, scale)
    with nRow number of rows and nCol number of columns

    :param nRow: int
    :param nCol: int
    :param kwargs: loc, scale, etc. for random variable simulation
    :return: 2D array-like
    """
    simNumbers = norm.rvs(size=nRow*nCol, **kwargs)
    return simNumbers.reshape(nRow, nCol)


def dataSimulation(root, numOfDraws):
    columns = np.shape(root)[1]
    stdNormal = multiNormalSimulation(columns, numOfDraws)
    simulatedData = root @ stdNormal
    return simulatedData


def copulaSimulation(data, numOfDraws):
    matrixU = data.copy()
    paramsList = []
    for column in data.columns:
        df, loc, scale = mleT(data[column])
        paramsList.append([df, loc, scale])
        matrixU[column] = t.cdf(data[column], df=df, loc=loc, scale=scale)
    spearmanCorr = pd.DataFrame(spearmanr(matrixU)[0])
    root = Cholesky(spearmanCorr)
    stdNormal = multiNormalSimulation(len(matrixU.columns), numOfDraws)
    simulatedData = (root @ stdNormal).T
    simulatedReturns = pd.DataFrame()
    for i in range(len(simulatedData.columns)):
        simulatedU = norm.cdf(simulatedData.iloc[:, i])
        simulatedReturns[data.columns[i]] = \
            t.ppf(simulatedU, df=paramsList[i][0], loc=paramsList[i][1], scale=paramsList[i][2])
    return simulatedReturns