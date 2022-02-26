import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize

"""
to do: refactor to a class, can return simulated data
"""


def mleT(data):
    def negLogLikeForT(initialParams):
        df, mean, sigma = initialParams
        return -t(df=df, loc=mean, scale=sigma).logpdf(data).sum()
    initialParams = np.array([2, data.mean(), data.std()])
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2},
            {'type': 'ineq', 'fun': lambda x:  x[2]})
    df, mean, sigma = minimize(negLogLikeForT, x0=initialParams, constraints=cons).x
    return df, mean, sigma


def mleNormal(data):
    def negLogLikeForNormal(initialParams):
        mean, sigma = initialParams
        return -norm(loc=mean, scale=sigma).logpdf(data).sum()
    initialParams = np.array([data.mean(), data.std()])
    cons = ({'type': 'ineq', 'fun': lambda x:  x[1]})
    mean, sigma = minimize(negLogLikeForNormal, x0=initialParams, constraints=cons).x
    return mean, sigma