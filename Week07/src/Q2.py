# import pandas as pd
# import numpy as np
# from os.path import abspath, dirname
# from Week07.src.Q1 import BTAmericanDiv, findBTAmericanDivIvol
# from scipy.stats import norm
# from RiskManagementPackage.VaR import VaR_distribution
# from RiskManagementPackage.ES import ES_distribution
#
# np.random.seed(12345)
#
# path = dirname(dirname(abspath(__file__)))
# data = pd.read_csv(path + '/data/problem2.csv')
# portNames = data['Portfolio'].unique()
#
# currUnderlying = 164.85
# underlyings = list(range(130, 210))
# current = pd.to_datetime("2022-02-25")
# rf = 0.0025
# coupon = 0
# b = rf - coupon
# initVol = 0.5
# daysInYear = 365
# divAmounts = [1]
# divTime = pd.to_datetime("2022-03-15")
# divTimes = [(divTime - current).days]
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
#         iVol = findBTAmericanDivIvol(data['isCall'][i], currUnderlying, data['Strike'][i], ttm, rf, b, divAmounts,
#                                      divTimes, days, data['CurrentPrice'][i])
#         iVolList.append(iVol)
#     else:
#         iVolList.append(np.nan)
# data['iVol'] = iVolList
#
#
# ret = pd.read_csv(path + '/data/DailyReturn.csv')['AAPL']
# stockStd = ret.std()
#
# # calculate delta
# deltaList = []
# increment = 0.001
# for i in range(len(data)):
#     if data['Type'][i] == "Option":
#         days = (data['ExpirationDate'][i] - current).days
#         ttm = days / daysInYear
#         increPrice = BTAmericanDiv(data['isCall'][i], currUnderlying + increment, data['Strike'][i], ttm, rf, b,
#                           divAmounts, divTimes, data['iVol'][i], days)
#         delta = (increPrice - data['CurrentPrice'][i]) / increment
#         deltaList.append(delta)
#     else:
#         deltaList.append(1)
# data['delta'] = deltaList
#
# # calculate portfolio PV
# portPV = []
# for portName in portNames:
#     port = data[data['Portfolio'] == portName]
#     portPV.append((port['Holding'] * port['CurrentPrice']).sum())
#
# # calculate portfolio gradient
# portGradients = []
# for i in range(len(portNames)):
#     port = data[data['Portfolio'] == portNames[i]]
#     gradient = currUnderlying / portPV[i] * (port['Holding'] * port['delta']).sum()
#     portGradients.append(gradient)
#
#
# # calculate portfolio std
# portStd = []
# for portGradient in portGradients:
#     portStd.append(np.abs(portGradient) * stockStd)
#
#
# # simulate portfolio values
# numOfDraws = 10000
# portSimRets = pd.DataFrame()
# for i in range(len(portNames)):
#     portSimRets[portNames[i]] = norm.rvs(size=numOfDraws, loc=0, scale=portStd[i])
#
#
#
# alpha = 0.05
# predMean = []
# predVaR = []
# predES = []
# for i in range(len(portNames)):
#     mean = portSimRets[portNames[i]].mean()
#     var = portPV[i] * VaR_distribution(norm, alpha) * portStd[i] * np.sqrt(10)
#     es = portPV[i] * ES_distribution(norm, alpha, numOfDraws) * portStd[i] * np.sqrt(10)
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
