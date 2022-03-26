import numpy as np
from scipy.optimize import minimize


def findRisk(ret, cov, targetRet):
    def objective(initialParams):
        weights = initialParams
        return weights.T @ cov @ weights
    initialParams = np.array([1 / len(cov)] * len(cov))
    cons = ({'type': 'eq', 'fun': lambda x:  sum(x) - 1},
            {'type': 'eq', 'fun': lambda x:  (x * ret).sum() - targetRet})
    weights = minimize(objective, x0=initialParams, constraints=cons).x
    riskVar = np.sqrt(objective(weights))
    return riskVar

def getRisks(ret, cov, targetRets):
    riskVars = []
    for targetRet in targetRets:
        riskVar = findRisk(ret, cov, targetRet)
        riskVars.append(riskVar)
    return riskVars


def findSuperPortfolio(targetRets, riskVars, rf):
    maxSharpe = 0
    maxIndex = -1
    for i in range(len(targetRets)):
        sharpe = (targetRets[i] - rf) / riskVars[i]
        if sharpe > maxSharpe:
            maxSharpe = sharpe
            maxIndex = i
    return maxSharpe, maxIndex

