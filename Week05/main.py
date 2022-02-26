import pandas as pd
import numpy as np
from os.path import dirname, abspath
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import seaborn as sns
from RiskManagementPackage.VaR import VaR_raw, VaR_distribution
from RiskManagementPackage.ES import ES_raw, ES_distribution
from RiskManagementPackage.MLE import mleNormal, mleT
from RiskManagementPackage.CalculateReturn import return_calculate
from RiskManagementPackage.Simulation import copulaSimulation

np.random.seed(12345)

# Q1
alpha = 0.05
size = 10000

path = dirname(abspath(__file__))
data = pd.read_csv(path + '/data/problem1.csv')
data = np.array(data['x'].tolist())

mean, sigma = mleNormal(data)
varNormal = VaR_distribution(norm, alpha, loc=mean, scale=sigma)
esNormal = ES_distribution(norm, alpha, size, loc=mean, scale=sigma)
normalSimulation = norm.rvs(size=size, loc=mean, scale=sigma)
print("VaR with normal distribution: ", varNormal)
print("ES with normal distribution: ", esNormal)

df, mean, sigma = mleT(data)
varT = VaR_distribution(t, alpha, df=df, loc=mean, scale=sigma)
esT = ES_distribution(t, alpha, size, df=df, loc=mean, scale=sigma)
tSimulation = t.rvs(size=size, df=df, loc=mean, scale=sigma)
print("VaR with t distribution: ", varT)
print("ES with t distribution: ", esT)

fig, ax = plt.subplots(figsize=(16, 8))
sns.histplot(data, stat='density', ax=ax, label='Real', color='grey')
ax.axvline(x=-varNormal, linestyle='dashed', alpha=0.5)
ax.text(x=-varNormal, y=4, s='VaR Normal', alpha=0.7, color='#334f8d')
ax.axvline(x=-esNormal, linestyle='dashed', alpha=0.5)
ax.text(x=-esNormal, y=5, s="ES Normal", alpha=0.7, color='#334f8d')
ax.axvline(x=-varT, linestyle='dashed', alpha=0.5)
ax.text(x=-varT, y=6, s="VaR T", alpha=0.7, color='#334f8d')
ax.axvline(x=-esT, linestyle='dashed', alpha=0.5)
ax.text(x=-esT, y=7, s="ES T", alpha=0.7, color='#334f8d')
sns.kdeplot(normalSimulation, ax=ax, label='Normal', color='red')
sns.kdeplot(tSimulation, ax=ax, label='T', color='blue')
ax.set_title("Distribution of Problem 1 Data")
ax.legend()
plt.show()


# Q3
alpha = 0.05
numOfDraws = 10000
path = dirname(abspath(__file__))
prices = pd.read_csv(path + '/data/DailyPrices.csv')
portfolios = pd.read_csv(path + '/data/portfolio.csv')
portfolioList = [['A'], ['B'], ['C'], ['A', 'B', 'C']]
historicVaRList = []
historicESList = []
monteCarloVaRList = []
monteCarloESList = []
for name in portfolioList:
    data = portfolios[portfolios['Portfolio'].isin(name)]
    holdings = data['Holding'].tolist()
    stockList = data['Stock'].tolist()
    data = data.set_index('Stock')
    portfolioPrices = prices[stockList]
    portfolioPrices = pd.concat([prices['Date'], portfolioPrices], axis=1)
    currentPrices = portfolioPrices.iloc[len(portfolioPrices) - 1, :]
    portfolioReturns = return_calculate(portfolioPrices, dateColumn="Date")
    portfolioReturns = portfolioReturns.iloc[:, 1:]
    simulatedReturns = copulaSimulation(portfolioReturns, numOfDraws)

    data = data.join(currentPrices[data.index])
    data.columns = ['Portfolio', 'Holding', 'Prices']
    data['Value'] = data['Prices'] * data['Holding']
    profitLoss = simulatedReturns * data['Value']
    portfolioLoss = profitLoss.sum(axis=1)

    historicVaRList.append(VaR_raw(portfolioLoss, alpha))
    historicESList.append(ES_raw(portfolioLoss, alpha))
    df, mean, sigma = mleT(portfolioLoss)
    monteCarloVaRList.append(VaR_distribution(t, alpha, df=df, loc=mean, scale=sigma))
    monteCarloESList.append(ES_distribution(t, alpha, numOfDraws, df=df, loc=mean, scale=sigma))

result = pd.DataFrame()
result['historicVaR'] = historicVaRList
result['monteCarloVaR'] = monteCarloVaRList
result['historicES'] = historicESList
result['monteCarloES'] = monteCarloESList
result.index = ['A', 'B', 'C', 'ABC']
print(result)