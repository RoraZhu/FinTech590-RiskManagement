import pandas as pd
import numpy as np
from os.path import abspath, dirname
from Week07.src.Q1 import BS, BTAmericanDiv, findBTAmericanDivIvol
from Week07.src.Q3 import getRisks, findSuperPortfolio, findRisk
from scipy.stats import norm
from RiskManagementPackage.VaR import VaR_distribution
from RiskManagementPackage.ES import ES_distribution
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

# Problem 1
underlying = 165
strike = 165
current = pd.to_datetime("2022-03-13")
expire = pd.to_datetime("2022-04-15")
rf = 0.0025
coupon = 0.0053
b = rf - coupon
days = (expire - current).days
ttm = days / 365
iVol = 0.2
N = (expire - current).days
divAmounts = [0.88]
divTimes = [(pd.to_datetime("2022-04-11") - current).days]

call = BS(True, underlying, strike, ttm, rf, b)
put = BS(False, underlying, strike, ttm, rf, b)

GBSM_call = [call.cal_delta(iVol), call.cal_gamma(iVol), call.cal_vega(iVol), call.cal_theta(iVol), np.nan,
             call.cal_carry_rho(iVol)]

GBSM_put = [put.cal_delta(iVol), put.cal_gamma(iVol), put.cal_vega(iVol), put.cal_theta(iVol), np.nan,
            put.cal_carry_rho(iVol)]

delta = 0.0001
finite_diff_call = []
# delta
new_call = BS(True, underlying+delta, strike, ttm, rf, b)
finite_diff_call.append((new_call.cal_val(iVol) - call.cal_val(iVol)) / delta)
# gamma
new_call1 = BS(True, underlying-delta, strike, ttm, rf, b)
finite_diff_call.append((new_call.cal_val(iVol) + new_call1.cal_val(iVol) - 2 * call.cal_val(iVol)) / (delta ** 2))
# vega
finite_diff_call.append((new_call.cal_val(iVol+delta) - call.cal_val(iVol)) / delta)
# theta
new_call = BS(True, underlying, strike, ttm+delta, rf, b)
finite_diff_call.append((new_call.cal_val(iVol) - call.cal_val(iVol)) / delta)
# rho
new_call = BS(True, underlying, strike, ttm, rf+delta, b)
finite_diff_call.append((new_call.cal_val(iVol) - call.cal_val(iVol)) / delta)
# carry rho
new_call = BS(True, underlying, strike, ttm, rf, b+delta)
finite_diff_call.append((new_call.cal_val(iVol) - call.cal_val(iVol)) / delta)

finite_diff_put = []
# delta
new_put = BS(False, underlying+delta, strike, ttm, rf, b)
finite_diff_put.append((new_put.cal_val(iVol) - put.cal_val(iVol)) / delta)
# gamma
new_put1 = BS(False, underlying-delta, strike, ttm, rf, b)
finite_diff_put.append((new_put.cal_val(iVol) + new_put1.cal_val(iVol) - 2 * put.cal_val(iVol)) / (delta ** 2))
# vega
finite_diff_put.append((new_put.cal_val(iVol+delta) - put.cal_val(iVol)) / delta)
# theta
new_put = BS(False, underlying, strike, ttm+delta, rf, b)
finite_diff_put.append((new_put.cal_val(iVol) - put.cal_val(iVol)) / delta)
# rho
new_put = BS(False, underlying, strike, ttm, rf+delta, b)
finite_diff_put.append((new_put.cal_val(iVol) - put.cal_val(iVol)) / delta)
# carry rho
new_put = BS(False, underlying, strike, ttm, rf, b+delta)
finite_diff_put.append((new_put.cal_val(iVol) - put.cal_val(iVol)) / delta)

results = pd.DataFrame()
results['GBSM_call'] = GBSM_call
results['finite_diff_call'] = finite_diff_call
results['GBSM_put'] = GBSM_put
results['finite_diff_put'] = finite_diff_put
results.index = ['delta', 'gamma', 'vega', 'theta', 'rho', 'carry_rho']
print(results)

# with dividend
call_val = BTAmericanDiv(True, underlying, strike, ttm, rf, b, divAmounts, divTimes, iVol, N)
put_val = BTAmericanDiv(False, underlying, strike, ttm, rf, b, divAmounts, divTimes, iVol, N)
print("Price of American call with dividend: ", call_val)
print("Price of American put with dividend: ", put_val)

delta = 0.0001
finite_diff_call = []
# delta
new_call_val = BTAmericanDiv(True, underlying+delta, strike, ttm, rf, b, divAmounts, divTimes, iVol, N)
finite_diff_call.append((new_call_val - call_val) / delta)
# gamma
new_call_val = BTAmericanDiv(True, underlying+1, strike, ttm, rf, b, divAmounts, divTimes, iVol, N)
new_call1_val = BTAmericanDiv(True, underlying-1, strike, ttm, rf, b, divAmounts, divTimes, iVol, N)
finite_diff_call.append((new_call_val + new_call1_val - 2 * call_val) / (1 ** 2))
# vega
new_call_val = BTAmericanDiv(True, underlying, strike, ttm, rf, b, divAmounts, divTimes, iVol+delta, N)
finite_diff_call.append((new_call_val - call_val) / delta)
# theta
new_call_val = BTAmericanDiv(True, underlying, strike, ttm+delta, rf, b, divAmounts, divTimes, iVol, N)
finite_diff_call.append((new_call_val - call_val) / delta)
# rho
new_call_val = BTAmericanDiv(True, underlying, strike, ttm, rf+delta, b, divAmounts, divTimes, iVol, N)
finite_diff_call.append((new_call_val - call_val) / delta)
# div amount
new_call_val = BTAmericanDiv(True, underlying, strike, ttm, rf, b, [divAmounts[0]+delta], divTimes, iVol, N)
finite_diff_call.append((new_call_val - call_val) / delta)

finite_diff_put = []
# delta
new_put_val = BTAmericanDiv(False, underlying+delta, strike, ttm, rf, b, divAmounts, divTimes, iVol, N)
finite_diff_put.append((new_put_val - put_val) / delta)
# gamma
new_put1_val = BTAmericanDiv(False, underlying+1, strike, ttm, rf, b, divAmounts, divTimes, iVol, N)
new_put1_val = BTAmericanDiv(False, underlying-1, strike, ttm, rf, b, divAmounts, divTimes, iVol, N)
finite_diff_put.append((new_put_val + new_put1_val - 2 * put_val) / (1 ** 2))
# vega
new_put_val = BTAmericanDiv(False, underlying, strike, ttm, rf, b, divAmounts, divTimes, iVol+delta, N)
finite_diff_put.append((new_put_val - put_val) / delta)
# theta
new_put_val = BTAmericanDiv(False, underlying, strike, ttm+delta, rf, b, divAmounts, divTimes, iVol, N)
finite_diff_put.append((new_put_val - put_val) / delta)
# rho
new_put_val = BTAmericanDiv(False, underlying, strike, ttm, rf+delta, b, divAmounts, divTimes, iVol, N)
finite_diff_put.append((new_put_val - put_val) / delta)
# div amount
new_put_val = BTAmericanDiv(False, underlying, strike, ttm, rf, b+delta, [divAmounts[0]+delta], divTimes, iVol, N)
finite_diff_put.append((new_put_val - put_val) / delta)

results = pd.DataFrame()
results['AmericanCallWithDiv'] = finite_diff_call
results['AmericanPutWithDiv'] = finite_diff_put
results.index = ['delta', 'gamma', 'vega', 'theta', 'rho', 'div_amount']
print(results)


# Problem 2
np.random.seed(12345)

path = dirname(abspath(__file__))
data = pd.read_csv(path + '/data/problem2.csv')
portNames = data['Portfolio'].unique()

currUnderlying = 164.85
underlyings = list(range(130, 210))
current = pd.to_datetime("2022-02-25")
rf = 0.0025
coupon = 0
b = rf - coupon
initVol = 0.5
daysInYear = 365
divAmounts = [1]
divTime = pd.to_datetime("2022-03-15")
divTimes = [(divTime - current).days]

data['ExpirationDate'] = pd.to_datetime(data['ExpirationDate'])
data['isCall'] = np.where((data['OptionType'] == "Call"), True, np.where((data['OptionType'] == "Put"), False, np.nan))

# Calculate iVol
iVolList = []
for i in range(len(data)):
    if data['Type'][i] == "Option":
        days = (data['ExpirationDate'][i] - current).days
        ttm = days / daysInYear
        iVol = findBTAmericanDivIvol(data['isCall'][i], currUnderlying, data['Strike'][i], ttm, rf, b, divAmounts,
                                     divTimes, days, data['CurrentPrice'][i])
        iVolList.append(iVol)
    else:
        iVolList.append(np.nan)
data['iVol'] = iVolList


ret = pd.read_csv(path + '/data/DailyReturn.csv')['AAPL']
stockStd = ret.std()

# calculate delta
deltaList = []
increment = 0.001
for i in range(len(data)):
    if data['Type'][i] == "Option":
        days = (data['ExpirationDate'][i] - current).days
        ttm = days / daysInYear
        increPrice = BTAmericanDiv(data['isCall'][i], currUnderlying + increment, data['Strike'][i], ttm, rf, b,
                          divAmounts, divTimes, data['iVol'][i], days)
        delta = (increPrice - data['CurrentPrice'][i]) / increment
        deltaList.append(delta)
    else:
        deltaList.append(1)
data['delta'] = deltaList

# calculate portfolio PV
portPV = []
for portName in portNames:
    port = data[data['Portfolio'] == portName]
    portPV.append((port['Holding'] * port['CurrentPrice']).sum())

# calculate portfolio gradient
portGradients = []
for i in range(len(portNames)):
    port = data[data['Portfolio'] == portNames[i]]
    gradient = currUnderlying / portPV[i] * (port['Holding'] * port['delta']).sum()
    portGradients.append(gradient)


# calculate portfolio std
portStd = []
for portGradient in portGradients:
    portStd.append(np.abs(portGradient) * stockStd)


# simulate portfolio values
numOfDraws = 10000
portSimRets = pd.DataFrame()
for i in range(len(portNames)):
    portSimRets[portNames[i]] = norm.rvs(size=numOfDraws, loc=0, scale=portStd[i]*np.sqrt(10))


alpha = 0.05
predMean = []
predVaR = []
predES = []
for i in range(len(portNames)):
    var = portPV[i] * VaR_distribution(norm, alpha) * portStd[i] * np.sqrt(10)
    es = portPV[i] * ES_distribution(norm, alpha, numOfDraws) * portStd[i] * np.sqrt(10)
    predVaR.append(var)
    predES.append(es)

results = pd.DataFrame()
results['Mean'] = ((1+portSimRets) * portPV).mean().tolist()
results['VaR'] = predVaR
results['ES'] = predES
results.index = portNames
print(results)


# Problem 3
path = dirname(abspath(__file__))
rets = pd.read_csv(path + '/data/DailyReturn.csv')
ff3 = pd.read_csv(path + '/data/F-F_Research_Data_Factors_daily.csv')
momentum = pd.read_csv(path + '/data/F-F_Momentum_Factor_daily.csv')

momentum.columns = ['Date', 'MOM']
ff3.columns = ["Date", "Mkt_RF", "SMB", "HML", "RF"]
ff = pd.merge(ff3, momentum, on="Date")
ff['Date'] = pd.to_datetime(ff['Date'], format="%Y%m%d")
ff = ff[ff['Date'] > "2012-01-01"].reset_index(drop=True)
ff[["Mkt_RF", "SMB", "HML", "MOM", "RF"]] = ff[["Mkt_RF", "SMB", "HML", "MOM", "RF"]] / 100

rets['Date'] = pd.to_datetime(rets['Date'])
rets = pd.merge(rets, ff, on="Date")
rets = rets.rename(columns={"BRK-B": "BRK_B"})

stocks = ['AAPL', 'FB', 'UNH', 'MA', 'MSFT', 'NVDA', 'HD', 'PFE', 'AMZN', 'BRK_B',
          'PG', 'XOM', 'TSLA', 'JPM', 'V', 'DIS', 'GOOGL', 'JNJ', 'BAC', 'CSCO']

params_list = []
for stock in stocks:
    rets[stock] = rets[stock] - rets['RF']
    formula = stock + "~Mkt_RF + SMB + HML + MOM"
    result = sm.ols(formula=formula, data=rets).fit()
    params_list.append(result.params[1:])

expected_rets = []
expected_rf = ff['RF'].mean()
expected_factors = ff[["Mkt_RF", "SMB", "HML", "MOM"]].mean(axis=0)
for i in range(len(params_list)):
    expected_ret = (np.log((params_list[i] * expected_factors).sum() + 1) + expected_rf) * 255
    expected_rets.append(expected_ret)
covariance = (np.log(rets[stocks] + 1)).cov() * 255

expReturns = pd.DataFrame()
expReturns['stock'] = stocks
expReturns['expected_annual_return'] = expected_rets
print(expReturns)
print(covariance)



rf = 0.0025
targetReturns = np.arange(0.01, 0.20, 0.002)
riskVars = getRisks(expected_rets, covariance, targetReturns)
maxSharpe, maxIndex = findSuperPortfolio(targetReturns, riskVars, rf)

plt.plot(riskVars, targetReturns)
plt.plot(riskVars[maxIndex], targetReturns[maxIndex], 'ro')
plt.show()

print("best optimal portfolio: ")
print("return: ", targetReturns[maxIndex])
print("volatility: ", riskVars[maxIndex])
print("sharpe: ", maxSharpe)