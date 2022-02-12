import numpy as np
from scipy.stats import norm


def deltaNormalVaR(alpha, rets, currPrices, holdings, dateColumn='Date'):
    rets = rets.drop(dateColumn, axis=1)
    presentValue = (currPrices * holdings).sum()
    weights = currPrices * holdings / presentValue
    sigma = np.cov(rets.T)
    portfolioSigma = np.sqrt(weights.T @ sigma @ weights)
    return -presentValue * norm.ppf(alpha) * portfolioSigma


def historicVaR(alpha, rets, currPrices, holdings, dateColumn='Date'):
    rets = rets.drop(dateColumn, axis=1)
    simulatedPrices = (1 + rets) * currPrices
    simulatedValues = simulatedPrices @ holdings
    simulatedValues = simulatedValues.sort_values()
    simulatedValues = simulatedValues.reset_index(drop=True)
    presentValue = (currPrices * holdings).sum()
    return presentValue - simulatedValues[int(alpha*len(rets))]