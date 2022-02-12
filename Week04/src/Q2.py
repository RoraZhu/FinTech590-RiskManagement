import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize
from Week03.src.Q1 import exponentialWeights, expCovForPair


# date, price1, price2, ....
# date from oldest to youngest
def return_calculate(prices, method="DISCRETE", dateColumn="date"):
    if dateColumn not in prices.columns:
        print("dateColumn: ", dateColumn, " not in DataFrame: ", prices)
        return
    rets = prices.copy()
    if method.upper() == "DISCRETE":
        for column in prices.columns[1:]:
            rets[column] = prices[column] / prices[column].shift() - 1
    elif method.upper() == "LOG":
        for column in prices.columns[1:]:
            rets[column] = np.log(prices[column] / prices[column].shift())
    else:
        print("method: ", method, " must be DISCRETE or LOG")
    return rets.iloc[1:, :]


def normalVaR(alpha, mu, sigma):
    return -norm.ppf(alpha, loc=mu, scale=sigma)


def EWNormalVaR(alpha, lam, ret):
    weight = exponentialWeights(len(ret), lam)
    sigma = np.sqrt(expCovForPair(weight, ret, ret))
    mu = np.mean(ret)
    return normalVaR(alpha, mu, sigma)


def mleTVaR(alpha, ret):
    def negLogLikeForT(initialParams):
        df, sigma = initialParams
        return -t(df=df, scale=sigma).logpdf(ret).sum()
    initialParams = np.array([1, 1])
    df, sigma = minimize(negLogLikeForT, initialParams, method="BFGS").x
    return -t.ppf(alpha, df, loc=ret.mean(), scale=sigma)


def historicVaR(rets, alpha):
    return -np.quantile(rets, q=alpha)