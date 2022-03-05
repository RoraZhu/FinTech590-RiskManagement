# import pandas as pd
# import numpy as np
# from os.path import abspath, dirname
# from Week06.src.Q1 import BS
#
# path = dirname(dirname(abspath(__file__)))
# data = pd.read_csv(path + '/data/AAPL_Options.csv')
# data['Expiration'] = pd.to_datetime(data['Expiration'])
# data['isCall'] = np.where((data['Type'] == "Call"), True, False)
#
# underlying = 164.85
# current = pd.to_datetime("2022-02-25")
# rf = 0.0025
# coupon = 0.0053
# b = rf - coupon
# initVol = 0.5
# daysInYear = 365
#
# for i in range(len(data)):
#     days = (data['Expiration'][i] - current).days
#     ttm = days / daysInYear
#     option = BS(data['isCall'][i], underlying, data['Strike'][i], ttm, rf, b, data['Last Price'][i])
#     root = option.cal_ivol(initVol)
#     print("Type: ", data['Type'][i], " Strike: ", data['Strike'][i], " Root: ", root)
#
