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
print("OLS: ")
print(OLS.params)
print('\n')

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
def negLogLike(parameters):
    const, beta, std_dev = parameters
    mu = const + beta * data['x']
    return -1 * stats.norm(mu, std_dev).logpdf(data['y']).sum()


params = np.array([1, 1, 1])
res = minimize(negLogLike, params, method='BFGS')
data['mleResidualNormal'] = data['y'] - (res.x[0] + res.x[1] * data['x'])
print("MLE with normal assumption: ")
print("fitted intercept: ", res.x[0])
print("fitted beta: ", res.x[1])
print('\n')


# MLE with t statistics assumption
def negTDistribute(parameters):
    const, beta = parameters
    e = data['y'] - const - beta * data['x']
    return -stats.t(df=len(data)-2).logpdf(e).sum()
    # return np.log((len(data)-2) + (data['y'] - const - beta * data['x'])**2).sum()


params = np.array([1, 1])
res = minimize(negTDistribute, params, method='BFGS')
data['mleResidualT'] = data['y'] - (res.x[0] + res.x[1] * data['x'])
print("MLE with t assumption: ")
print("fitted intercept: ", res.x[0])
print("fitted beta: ", res.x[1])


# goodness of fit
ssTotal = ((data['y'] - np.mean(data['y']))**2).sum()
ssErrorNormal = (data['mleResidualNormal']**2).sum()
ssErrorT = (data['mleResidualT']**2).sum()
rSquareNormal = 1 - ssErrorNormal/ssTotal
rSquareT = 1 - ssErrorT/ssTotal

print("R square for MLE with normal distributed errors: ", rSquareNormal)
print("R square for MLE with t distributed errors: ", rSquareT)