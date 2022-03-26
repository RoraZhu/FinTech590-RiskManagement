import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import pandas as pd


class BS:
    def __init__(self, isCall, underlying, strike, ttm, rf, b, price=0):
        self.isCall = isCall
        self.underlying = underlying
        self.strike = strike
        self.ttm = ttm
        self.rf = rf
        self.b = b
        self.price = price

    def _cal_d1d2(self, iVol):
        d1 = (np.log(self.underlying / self.strike) + (self.b + iVol ** 2 / 2) * self.ttm) / (iVol * np.sqrt(self.ttm))
        d2 = d1 - iVol * np.sqrt(self.ttm)
        return d1, d2

    def _callValue(self, d1, d2):
        return self.underlying * np.exp((self.b - self.rf) * self.ttm) * norm.cdf(d1) - self.strike * np.exp(
            -self.rf * self.ttm) * norm.cdf(d2)

    def _putValue(self, d1, d2):
        return self.strike * np.exp(-self.rf * self.ttm) * norm.cdf(-d2) - self.underlying * np.exp(
            (self.b - self.rf) * self.ttm) * norm.cdf(-d1)

    def _timeValue(self):
        return np.exp((self.b - self.rf) * self.ttm)

    def _iVol(self, iVol):
        return self.cal_val(iVol) - self.price

    def cal_iVol(self, initVol):
        root = fsolve(self._iVol, initVol)
        return root[0]

    def cal_val(self, iVol):
        d1, d2 = self._cal_d1d2(iVol)
        if self.isCall:
            return self._callValue(d1, d2)
        else:
            return self._putValue(d1, d2)

    def cal_delta(self, iVol):
        d1, d2 = self._cal_d1d2(iVol)
        sign = 0 if self.isCall else 1
        return self._timeValue() * (norm.cdf(d1) - sign)

    def cal_gamma(self, iVol):
        d1, d2 = self._cal_d1d2(iVol)
        return norm.pdf(d1) * self._timeValue() / (self.underlying * iVol * np.sqrt(self.ttm))

    def cal_vega(self, iVol):
        d1, d2 = self._cal_d1d2(iVol)
        return self.underlying * self._timeValue() * norm.pdf(d1) * np.sqrt(self.ttm)

    def cal_theta(self, iVol):
        d1, d2 = self._cal_d1d2(iVol)
        term1 = (self.underlying * self._timeValue() * norm.pdf(d1) * iVol) / (2 * np.sqrt(self.ttm))
        term2 = (self.b - self.rf) * self.underlying * self._timeValue()
        term3 = self.rf * self.underlying * np.exp(-self.rf * self.ttm)
        sign = 1 if self.isCall else -1
        return term1 + sign * term2 * norm.cdf(sign * d1) + sign * term3 * norm.cdf(sign * d2)

    def cal_carry_rho(self, iVol):
        d1, d2 = self._cal_d1d2(iVol)
        sign = 1 if self.isCall else -1
        return sign * self.ttm * self.underlying * self._timeValue() * norm.cdf(sign * d1)


def BTAmerican(isCall, underlying, strike, ttm, rf, b, iVol, N):
    dt = ttm / N
    u = np.exp(iVol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1 - pu
    df = np.exp(-rf * dt)
    z = 1 if isCall else -1

    def indexFunc(i, j):
        return int(j * (j + 1) / 2 + i)

    nNodes = indexFunc(0, N + 1)
    optionValues = [0] * nNodes

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            index = indexFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))
            optionValues[index] = max(0, z * (price - strike))
            if j < N:
                pv = df * (pu * optionValues[indexFunc(i + 1, j + 1)] + pd * optionValues[indexFunc(i, j + 1)])
                optionValues[index] = max(optionValues[index], pv)
    return optionValues[0]


def BTAmericanDiv(isCall, underlying, strike, ttm, rf, b, divAmounts, divTimes, iVol, N):
    if not divAmounts or not divTimes or divTimes[0] > N:
        return BTAmerican(isCall, underlying, strike, ttm, rf, b, iVol, N)

    def indexFunc(i, j):
        return int(j * (j + 1) / 2 + i)

    dt = ttm / N
    u = np.exp(iVol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp((- rf) * dt)
    z = 1 if isCall else -1

    nDiv = len(divTimes)
    nNodes = indexFunc(0, divTimes[0] + 1)
    optionValues = [0] * nNodes

    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            index = indexFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))
            if j < divTimes[0]:
                optionValues[index] = max(0, z * (price - strike))
                pv = df * (pu * optionValues[indexFunc(i + 1, j + 1)] + pd * optionValues[indexFunc(i, j + 1)])
                optionValues[index] = max(optionValues[index], pv)

            else:
                valNoExercise = BTAmericanDiv(
                    isCall, price - divAmounts[0], strike, ttm - divTimes[0] * dt,
                    rf, b, divAmounts[1:], [divTimes[i] - divTimes[0] for i in range(1, nDiv)], iVol, N - divTimes[0]
                )
                valExercise = max(0, z * (price - strike))
                optionValues[index] = max(valNoExercise, valExercise)

    return optionValues[0]


def findBTAmericanDivIvol(isCall, underlying, strike, ttm, rf, b, divAmounts, divTimes, N, price):
    def objective(iVol, isCall, underlying, strike, ttm, rf, b, divAmounts, divTimes, N):
        return BTAmericanDiv(isCall, underlying, strike, ttm, rf, b, divAmounts, divTimes, iVol, N) - price
    initIVol = np.array([0.5])
    root = fsolve(objective, initIVol, args=(isCall, underlying, strike, ttm, rf, b, divAmounts, divTimes, N))
    return root[0]

