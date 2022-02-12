import pandas as pd
import numpy as np
from os.path import dirname, abspath
import matplotlib.pyplot as plt
import yfinance as yf
import Week04.src.Q1 as Q1
import Week04.src.Q2 as Q2
import Week04.src.Q3 as Q3


# Question 1
np.random.seed(12345)

P0 = 100
mu = 0
sigma = 0.1
rets = np.random.normal(mu, sigma, 10000)
brownianP1 = Q1.brownianPrice(P0, rets)
arithmeticP1 = Q1.arithmeticPrice(P0, rets)
geometricP1 = Q1.geometricBrownPrice(P0, rets)

mean = []
std = []
for prices in [brownianP1, arithmeticP1, geometricP1]:
    mean.append(np.mean(prices))
    std.append(np.std(prices))

summary = pd.DataFrame()
summary['mean'] = mean
summary['std'] = std
summary.index = ['brownian', 'arithmatic', 'geometric']
print(summary, '\n')

# Question 2
path = dirname(abspath(__file__))
prices = pd.read_csv(path + '/data/DailyPrices.csv')
rets = Q2.return_calculate(prices, method="DISCRETE", dateColumn="Date")

alpha = 0.05
lam = 0.94
intcRets = rets['INTC']
intcRets = intcRets - intcRets.mean()
intcMean = np.mean(intcRets)
intcStd = np.std(intcRets)

intcNormalVaR = Q2.normalVaR(alpha, intcMean, intcStd)
intcEWNormalVaR = Q2.EWNormalVaR(alpha, lam, intcRets)
intcMleTVaR = Q2.mleTVaR(alpha, intcRets)
intcHistoricVaR = Q2.historicVaR(intcRets, alpha)
print("INTC normal VaR: ", round(intcNormalVaR*100, 2), "%")
print("INTC EW normal VaR: ", round(intcEWNormalVaR*100, 2), "%")
print("INTC MLE with T VaR: ", round(intcMleTVaR*100, 2), "%")
print("INTC historical VaR: ", round(intcHistoricVaR*100, 2), "%", '\n')

outOfSamplePrice = yf.download('INTC', start='2022-01-15', end='2022-02-11', progress=False)
outOfSamplePrice['Date'] = outOfSamplePrice.index
outOfSamplePrice = outOfSamplePrice[['Date', 'Adj Close']]
outOfSampleReturn = Q2.return_calculate(outOfSamplePrice, method="DISCRETE", dateColumn="Date")

plt.hist(intcRets, label='in sample')
plt.hist(outOfSampleReturn['Adj Close'], label='out of sample')
plt.legend(loc='upper left')
plt.show()


# Question 3
prices = pd.read_csv(path + '/data/DailyPrices.csv')
portfolios = pd.read_csv(path + '/data/portfolio.csv')

alpha = 0.05
portfolioList = [['A'], ['B'], ['C'], ['A', 'B', 'C']]
historic = []
deltaNormal = []
for name in portfolioList:
    data = portfolios[portfolios['Portfolio'].isin(name)]
    holdings = data['Holding'].tolist()
    stockList = data['Stock'].tolist()
    portfolioPrices = prices[stockList]
    portfolioPrices = pd.concat([prices['Date'], portfolioPrices], axis=1)
    portfolioReturns = Q2.return_calculate(portfolioPrices, dateColumn='Date')
    currentPrices = portfolioPrices.iloc[len(portfolioPrices)-1, 1:]
    historic.append(Q3.historicVaR(alpha, portfolioReturns, currentPrices, holdings, 'Date'))
    deltaNormal.append(Q3.deltaNormalVaR(alpha, portfolioReturns, currentPrices, holdings))

summary = pd.DataFrame()
summary['historicVaR'] = historic
summary['deltaNormalVaR'] = deltaNormal
summary.index = ['A', 'B', 'C', 'Total']
print(summary)
