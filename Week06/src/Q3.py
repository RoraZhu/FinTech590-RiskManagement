import pandas as pd
import numpy as np
from os.path import abspath, dirname
from scipy.stats import norm
from Week06.src.Q1 import BS
from RiskManagementPackage.VaR import VaR_raw
from RiskManagementPackage.ES import ES_raw

# np.random.seed(12345)
#
# path = dirname(dirname(abspath(__file__)))
# data = pd.read_csv(path + '/data/problem3.csv')
#
# currUnderlying = 164.85
# underlyings = list(range(130, 210))
# current = pd.to_datetime("2022-02-25")
# rf = 0.0025
# coupon = 0.0053
# b = rf - coupon
# initVol = 0.5
# daysInYear = 365
#
# data['ExpirationDate'] = pd.to_datetime(data['ExpirationDate'])
# data['isCall'] = np.where((data['OptionType'] == "Call"), True, np.where((data['OptionType'] == "Put"), False, np.nan))
#
# # Calculate iVol
# iVolList = []
# for i in range(len(data)):
#     if data['Type'][i] == "Option":
#         days = (data['ExpirationDate'][i] - current).days
#         ttm = days / daysInYear
#         option = BS(data['isCall'][i], currUnderlying, data['Strike'][i], ttm, rf, b, data['CurrentPrice'][i])
#         iVol = option.cal_ivol(initVol)
#         iVolList.append(iVol)
#     else:
#         iVolList.append(np.nan)
# data['iVol'] = iVolList
#
# # Calculate option values
# values = pd.DataFrame()
# for i in range(len(data)):
#     if data['Type'][i] == "Option":
#         days = (data['ExpirationDate'][i] - current).days
#         ttm = days / daysInYear
#
#         valueList = []
#         for underlying in underlyings:
#             option = BS(data['isCall'][i], underlying, data['Strike'][i], ttm, rf, b)
#             valueList.append(option.cal_val(data['iVol'][i]))
#         values[data['Portfolio'][i] + "-" + data['OptionType'][i] + str(data['Strike'][i])] = valueList
#
#     else:
#         values[data['Portfolio'][i] + "-" + "Stock"] = underlyings
#
# # Calculate portfolio values
# portValues = pd.DataFrame()
# portNames = data['Portfolio'].unique()
# for portName in portNames:
#     port = data[data['Portfolio'] == portName]
#     holdings = port['Holding'].to_list()
#     value = values.iloc[:, port.index]
#     portValues[portName] = (value * holdings).sum(axis=1)
#
# ret = pd.read_csv(path + '/data/DailyReturn.csv')['AAPL']
# size = 10
# numOfDraws = 10000
# scale = ret.std()
# sim_underlying = []
#
# for i in range(numOfDraws):
#     sim_ret = norm.rvs(size=size, loc=0, scale=ret.std())
#     sim_underlying.append(currUnderlying * (sim_ret + 1).prod())
#
# # Calculate option predicted values
# predValues = pd.DataFrame()
# for i in range(len(data)):
#     print(i)
#     if data['Type'][i] == "Option":
#         days = (data['ExpirationDate'][i] - current).days - 10
#         ttm = days / daysInYear
#         valueList = []
#         for underlying in sim_underlying:
#             option = BS(data['isCall'][i], underlying, data['Strike'][i], ttm, rf, b)
#             valueList.append(option.cal_val(data['iVol'][i]))
#         predValues[data['Portfolio'][i] + "-" + data['OptionType'][i] + str(data['Strike'][i])] = valueList
#     else:
#         predValues[data['Portfolio'][i] + "-" + "Stock"] = sim_underlying
#
# # Calculate portfolio predicted values
# predPortValues = pd.DataFrame()
# portNames = data['Portfolio'].unique()
# for portName in portNames:
#     port = data[data['Portfolio'] == portName]
#     holdings = port['Holding'].to_list()
#     value = predValues.iloc[:, port.index]
#     predPortValues[portName] = (value * holdings).sum(axis=1)
# print(predPortValues)
#
# alpha = 0.05
# predMean = []
# predVaR = []
# predES = []
# for portName in portNames:
#     mean = predPortValues[portName].mean()
#     var = VaR_raw(predPortValues[portName], alpha)
#     es = ES_raw(predPortValues[portName], alpha)
#     predMean.append(mean)
#     predVaR.append(var)
#     predES.append(es)
#
# results = pd.DataFrame()
# results['Mean'] = predMean
# results['VaR'] = predVaR
# results['ES'] = predES
# results.index = portNames
# print(results)