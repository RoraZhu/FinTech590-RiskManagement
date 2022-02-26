from RiskManagementPackage.VaR import VaR_raw, VaR_distribution
import numpy as np


def ES_raw(data, alpha):
    data = sorted(data)
    var = - VaR_raw(data, alpha)
    return - np.mean([i for i in data if i <= var])


def ES_distribution(dist, alpha, size, **kwargs):
    var = - VaR_distribution(dist, alpha, **kwargs)
    numbers = dist.rvs(size=size, **kwargs)
    return - numbers[numbers < var].mean()