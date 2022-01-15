import pandas as pd
import numpy as np
from os.path import dirname, abspath
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro

path = dirname(dirname(abspath(__file__)))
data = pd.read_csv(path + '/problem2.csv')

# OLS
OLS = sm.ols(formula="y~x", data=data).fit()
data['olsResidual'] = OLS.resid

# check normality
# histogram
plt.hist(data['olsResidual'])
plt.show()

# qqplot
qqplot(data['olsResidual'], line='s')
plt.show()

# S-W test
stat, p = shapiro(data['olsResidual'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# MLE with normal assumption
def negLogLikeForNormal(parameters):
    const, beta, std_dev = parameters
    e = data['y'] - const - beta * data['x']
    return -1 * stats.norm(0, std_dev).logpdf(e).sum()


paramsNormal = np.array([1, 1, 1])
resNormal = minimize(negLogLikeForNormal, paramsNormal, method='BFGS')
data['mleResidualNormal'] = data['y'] - (resNormal.x[0] + resNormal.x[1] * data['x'])



# MLE with t statistics assumption
def negLogLikeForT(parameters):
    const, beta, df, scale = parameters
    e = data['y'] - const - beta * data['x']
    return -stats.t(df=df, scale=scale).logpdf(e).sum()


paramsT = np.array([1, 1, 1, 1])
resT = minimize(negLogLikeForT, paramsT, method='BFGS')
data['mleResidualT'] = data['y'] - (resT.x[0] + resT.x[1] * data['x'])


# goodness of fit
ssTotal = ((data['y'] - np.mean(data['y']))**2).sum()
ssErrorNormal = (data['mleResidualNormal']**2).sum()
ssErrorT = (data['mleResidualT']**2).sum()
rSquareNormal = 1 - ssErrorNormal/ssTotal
rSquareT = 1 - ssErrorT/ssTotal

# Information Criteria
aicNormal = 2 * len(resNormal.x) + 2 * negLogLikeForNormal(resNormal.x)
aicT = 2 * len(resT.x) + 2 * negLogLikeForT(resT.x)

bicNormal = 2 * negLogLikeForNormal(resNormal.x) + len(resNormal.x) * np.log(len(data))
bicT = 2 * negLogLikeForT(resT.x) + len(resT.x) * np.log(len(data))


print("R square for MLE with normal distributed errors: ", rSquareNormal)
print("R square for MLE with t distributed errors: ", rSquareT)

print("AIC for normal distributed error: ", aicNormal)
print("BIC for normal distributed error: ", bicNormal)
print("AIC for T distributed error: ", aicT)
print("BIC for T distributed error: ", bicT)

print("OLS parameters: ")
print(OLS.params)
print('\n')
print("MLE with normal assumption: ")
print("Intercept ", resNormal.x[0])
print("beta ", resNormal.x[1])
print('\n')
print("MLE with t assumption: ")
print("Intercept ", resT.x[0])
print("beta ", resT.x[1])