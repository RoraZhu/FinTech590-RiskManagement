import pandas as pd
import numpy as np
from os.path import dirname, abspath
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

path = dirname(dirname(abspath(__file__)))
data = pd.read_csv(path + '/problem1.csv')

# construct vector x and vector y
vectorX = data.iloc[:, 0]
vectorY = data.iloc[:, 1]

# calculate expectation of x and y
meanX = sum(vectorX)/len(vectorX)
meanY = sum(vectorY)/len(vectorY)

# calculate covariance matrix
covZ = np.cov(vectorX, vectorY, ddof=1)
covXX = covZ[0, 0]
covXY = covZ[0, 1]
covYX = covXY
covYY = covZ[1, 1]

# Expectation of Y given X
conExpY = pd.Series([meanY + covYX / covXX * (x - meanX) for x in vectorX])
print(conExpY)

# OLS
result = sm.ols(formula="y~x", data=data).fit()
fittedY = result.fittedvalues
print(result.summary())

# Expectation of Y given X equals to fitted Y in OLS
print(pd.DataFrame({'conExpY': conExpY, 'fittedY': fittedY}))

# Draw picture
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(vectorX, vectorY, 'o', label="raw data")
ax.plot(vectorX, conExpY, 'g--', label="Conditional Expected Y")
ax.plot(vectorX, fittedY, 'b--', label="OLS")
ax.legend(loc='best')
plt.show()

# reference
# https://stats.stackexchange.com/questions/71260/what-is-the-intuition-behind-conditional-gaussian-distributions
# https://stats.stackexchange.com/questions/71260/what-is-the-intuition-behind-conditional-gaussian-distributions
# https://online.stat.psu.edu/stat505/lesson/6/6.1
# Multivariate probability distributions and linear regression.pdf