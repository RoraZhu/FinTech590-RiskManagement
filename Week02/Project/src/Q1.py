import pandas as pd
import numpy as np
from os.path import dirname, abspath
import statsmodels.formula.api as sm

path = dirname(dirname(abspath(__file__)))
data = pd.read_csv(path + '/problem1.csv')

vectorX = data.iloc[:, 0]
vectorY = data.iloc[:, 1]
vectorZ = vectorX.append(vectorY, ignore_index=True)

# calculate mean
meanX = sum(vectorX)/len(vectorX)
meanY = sum(vectorY)/len(vectorY)
meanZ = [meanX, meanY]

# calculate covariance
covZ = np.cov(vectorX, vectorY, ddof=1)
covXX = covZ[0, 0]
covXY = covZ[0, 1]
covYX = covXY
covYY = covZ[1, 1]

# Expectation of Y given X
conExpY = pd.Series([meanY + covYX / covXX * (x - meanX) for x in vectorX])
conCovY = covYY - covYX / covXX * covXY

# OLS
result = sm.ols(formula="y~x", data=data).fit()
fittedY = result.fittedvalues

# Expectation of Y given X == fitted Y in OLS
print(pd.DataFrame({'conExpY': conExpY, 'fittedY': fittedY}))
