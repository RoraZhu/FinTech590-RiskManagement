import pandas as pd
import numpy as np
from Week06.src.Q1 import BS
from os.path import abspath, dirname
from scipy.stats import norm
from RiskManagementPackage.VaR import VaR_raw
from RiskManagementPackage.ES import ES_raw
import matplotlib.pyplot as plt


# Problem 1
strike = 165
underlying = 165
current = pd.to_datetime("2022-02-25")
expire = pd.to_datetime("2022-03-18")
rf = 0.0025
coupon = 0.0053
b = rf - coupon
days = (expire - current).days
ttm = days / 365

callList = []
putList = []
iVolList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for iVol in iVolList:
    call = BS(True, underlying, strike, ttm, rf, b)
    put = BS(False, underlying, strike, ttm, rf, b)
    callList.append(call.cal_val(iVol))
    putList.append(put.cal_val(iVol))
results = pd.DataFrame()
results['iVol'] = iVolList
results['Call'] = callList
results['Put'] = putList
print(results)

plt.plot(results['iVol'], results['Call'], label="Call")
plt.plot(results['iVol'], results['Put'], label="Put")
plt.xlabel('iVol')
plt.ylabel('Price')
plt.legend()
plt.show()


# Problem 2
path = dirname(abspath(__file__))
data = pd.read_csv(path + '/data/AAPL_Options.csv')
data['Expiration'] = pd.to_datetime(data['Expiration'])
data['isCall'] = np.where((data['Type'] == "Call"), True, False)

underlying = 164.85
current = pd.to_datetime("2022-02-25")
rf = 0.0025
coupon = 0.0053
b = rf - coupon
initVol = 0.5
daysInYear = 365

iVolList = []
for i in range(len(data)):
    days = (data['Expiration'][i] - current).days
    ttm = days / daysInYear
    option = BS(data['isCall'][i], underlying, data['Strike'][i], ttm, rf, b, data['Last Price'][i])
    iVolList.append(option.cal_ivol(initVol))
data['iVol'] = iVolList
print(data)

plt.scatter(data[data['Type'] == "Call"]['Strike'], data[data['Type'] == "Call"]['iVol'], label="Call")
plt.scatter(data[data['Type'] == "Put"]['Strike'], data[data['Type'] == "Put"]['iVol'], label="Put")
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.legend()
plt.show()


# Problem 3
np.random.seed(12345)

data = pd.read_csv(path + '/data/problem3.csv')

currUnderlying = 164.85
underlyings = list(range(130, 210))
current = pd.to_datetime("2022-02-25")
rf = 0.0025
coupon = 0.0053
b = rf - coupon
initVol = 0.5
daysInYear = 365

data['ExpirationDate'] = pd.to_datetime(data['ExpirationDate'])
data['isCall'] = np.where((data['OptionType'] == "Call"), True, np.where((data['OptionType'] == "Put"), False, np.nan))

# Calculate iVol
iVolList = []
for i in range(len(data)):
    if data['Type'][i] == "Option":
        days = (data['ExpirationDate'][i] - current).days
        ttm = days / daysInYear
        option = BS(data['isCall'][i], currUnderlying, data['Strike'][i], ttm, rf, b, data['CurrentPrice'][i])
        iVol = option.cal_ivol(initVol)
        iVolList.append(iVol)
    else:
        iVolList.append(np.nan)
data['iVol'] = iVolList

# Calculate option values
portNames = data['Portfolio'].unique()
values = pd.DataFrame()
for i in range(len(data)):
    if data['Type'][i] == "Option":
        days = (data['ExpirationDate'][i] - current).days
        ttm = days / daysInYear

        valueList = []
        for underlying in underlyings:
            option = BS(data['isCall'][i], underlying, data['Strike'][i], ttm, rf, b)
            valueList.append(option.cal_val(data['iVol'][i]))
        values[data['Portfolio'][i] + "-" + data['OptionType'][i] + str(data['Strike'][i])] = valueList

    else:
        values[data['Portfolio'][i] + "-" + "Stock"] = underlyings

# Calculate portfolio values
portValues = pd.DataFrame()
for portName in portNames:
    port = data[data['Portfolio'] == portName]
    holdings = port['Holding'].to_list()
    value = values.iloc[:, port.index]
    portValues[portName] = (value * holdings).sum(axis=1)


fig, axs = plt.subplots(3, 3, figsize=(12, 12))
for i in range(3):
    for j in range(3):
        axs[i][j].plot(underlyings, portValues.iloc[:, 3*i+j])
        axs[i][j].set_title(portNames[3*i+j])
        axs[i][j].set_xlabel("Underlying")
        axs[i][j].set_ylabel("Portfolio Value")
plt.show()


ret = pd.read_csv(path + '/data/DailyReturn.csv')['AAPL']
size = 10
numOfDraws = 10000
scale = ret.std()
sim_underlying = []

for i in range(numOfDraws):
    sim_ret = norm.rvs(size=size, loc=0, scale=ret.std())
    sim_underlying.append(currUnderlying * (sim_ret + 1).prod())

# Calculate option predicted values
predValues = pd.DataFrame()
for i in range(len(data)):
    if data['Type'][i] == "Option":
        days = (data['ExpirationDate'][i] - current).days - 10
        ttm = days / daysInYear
        valueList = []
        for underlying in sim_underlying:
            option = BS(data['isCall'][i], underlying, data['Strike'][i], ttm, rf, b)
            valueList.append(option.cal_val(data['iVol'][i]))
        predValues[data['Portfolio'][i] + "-" + data['OptionType'][i] + str(data['Strike'][i])] = valueList
    else:
        predValues[data['Portfolio'][i] + "-" + "Stock"] = sim_underlying

# Calculate portfolio predicted values
predPortValues = pd.DataFrame()
portNames = data['Portfolio'].unique()
for portName in portNames:
    port = data[data['Portfolio'] == portName]
    holdings = port['Holding'].to_list()
    value = predValues.iloc[:, port.index]
    predPortValues[portName] = (value * holdings).sum(axis=1)

# Calculate current portfolio values
currValues = []
for portName in portNames:
    port = data[data['Portfolio'] == portName]
    currValues.append((port['Holding'] * port['CurrentPrice']).sum())

predPortValuesDemean = predPortValues - np.array(currValues)

alpha = 0.05
predMean = []
predVaR = []
predES = []
for portName in portNames:
    mean = predPortValues[portName].mean()
    var = VaR_raw(predPortValuesDemean[portName], alpha)
    es = ES_raw(predPortValuesDemean[portName], alpha)
    predMean.append(mean)
    predVaR.append(var)
    predES.append(es)

results = pd.DataFrame()
results['Mean'] = predMean
results['VaR'] = predVaR
results['ES'] = predES
results.index = portNames
print(results)
