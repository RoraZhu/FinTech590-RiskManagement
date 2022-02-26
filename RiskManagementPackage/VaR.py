import numpy as np
from scipy.stats import norm, t
from RiskManagementPackage.CovarianceEstimation import getEWCov
from RiskManagementPackage.MLE import mleT, mleNormal


def VaR_raw(data, alpha):
    data = sorted(data)
    ceil = int(np.ceil(len(data) * alpha))
    floor = int(np.floor(len(data) * alpha))
    return - (data[ceil] + data[floor]) / 2


def VaR_distribution(dist, alpha,  **kwargs):
    return -dist.ppf(alpha, **kwargs)


def EWNormalVaR(alpha, lam, ret):
    sigma = np.sqrt(getEWCov(ret, ret, lam))
    mu = np.mean(ret)
    return VaR_distribution(norm, alpha, loc=mu, scale=sigma)


def mleNormalVaR(alpha, ret):
    mean, sigma = mleNormal(ret)
    return VaR_distribution(norm, alpha, loc=mean, scale=sigma)


def mleTVaR(alpha, ret):
    df, mean, sigma = mleT(ret)
    return VaR_distribution(t, alpha, df=df, loc=mean, scale=sigma)


def historicVaR(rets, alpha):
    return -np.quantile(rets, q=alpha)