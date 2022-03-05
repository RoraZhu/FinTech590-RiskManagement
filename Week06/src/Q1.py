import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


class BS:
    def __init__(self, isCall, underlying, strike, ttm, rf, b, price=0):
        self.isCall = isCall
        self.underlying = underlying
        self.strike = strike
        self.ttm = ttm
        self.rf = rf
        self.b = b
        self.price = price

    def _cal_d1d2(self, ivol):
        d1 = (np.log(self.underlying / self.strike) + (self.b + ivol ** 2 / 2) * self.ttm) / (ivol * np.sqrt(self.ttm))
        d2 = d1 - ivol * np.sqrt(self.ttm)
        return d1, d2

    def _callValue(self, d1, d2):
        return self.underlying * np.exp((self.b - self.rf) * self.ttm) * norm.cdf(d1) - self.strike * np.exp(
            -self.rf * self.ttm) * norm.cdf(d2)

    def _putValue(self, d1, d2):
        return self.strike * np.exp(-self.rf * self.ttm) * norm.cdf(-d2) - self.underlying * np.exp(
            (self.b - self.rf) * self.ttm) * norm.cdf(-d1)

    def _ivol(self, ivol):
        return self.cal_val(ivol) - self.price

    def cal_ivol(self, initVol):
        root = fsolve(self._ivol, initVol)
        return root[0]

    def cal_val(self, ivol):
        d1, d2 = self._cal_d1d2(ivol)
        if self.isCall:
            return self._callValue(d1, d2)
        else:
            return self._putValue(d1, d2)