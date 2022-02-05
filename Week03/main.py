import Week03.src.Q1 as Q1
import Week03.src.Q2 as Q2
import Week03.src.Q3 as Q3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath
import time

np.random.seed(12345)

path = dirname(abspath(__file__))
data = pd.read_csv(path + '/data/DailyReturn.csv')
data = data.iloc[:, 1:len(data.columns)]

# # Question 1
# lambdas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.97]
# for i in range(len(lambdas)):
#     print("lambda = ", lambdas[i])
#     covMatrix = Q1.expCovForFrame(data, lambdas[i])
#     cumEigen = Q1.pcaExplained(covMatrix)
#     plt.plot(cumEigen, label="lambda="+str(lambdas[i]))
# plt.legend()
# plt.show()

# Question 2
# executionTimeForHigham = []
# executionTimeForJacker = []
# frobeniusNormForHigham = []
# frobeniusNormForJacker = []
# for n in [10, 50, 100, 500, 1000, 2000]:
#     print(n)
#     A = [[0.9 for i in range(n)] for j in range(n)]
#     for i in range(len(A)):
#         A[i][i] = 1
#     A[0][1] = 0.7357
#     A[1][0] = 0.7357
#     A = pd.DataFrame(A)
#
#     startTime = time.time()
#     highamA = Q2.highamNearPSD(A, 100, 1e-9)
#     executionTimeForHigham.append(time.time() - startTime)
#
#     startTime = time.time()
#     nearA = Q2.nearPSD(A)
#     executionTimeForJacker.append(time.time() - startTime)
#
#     frobeniusNormForHigham.append(Q2.frobeniusNorm(highamA - A))
#     frobeniusNormForJacker.append(Q2.frobeniusNorm(nearA - A))
#
# valuation = pd.DataFrame()
# valuation['executionTimeForHigham'] = executionTimeForHigham
# valuation['executionTimeForJacker'] = executionTimeForJacker
# valuation['frobeniusNormForHigham'] = frobeniusNormForHigham
# valuation['frobeniusNormForJacker'] = frobeniusNormForJacker
# print(valuation)



# Question 3
lam = 0.97
explainedPowers = [1, 0.75, 0.5]
numOfDraws = 25000
executionTime = []
norms = []

pearsonCov = np.cov(data.T)
pearsonStd = Q3.generateStd(pearsonCov)
pearsonCorr = Q3.generateCorr(pearsonCov)

expCov = Q1.expCovForFrame(data, lam)
expStd = Q3.generateStd(expCov)
expCorr = Q3.generateCorr(expCov)

covPP = Q3.generateCov(pearsonStd, pearsonCorr)
covPE = Q3.generateCov(pearsonStd, expCorr)
covEP = Q3.generateCov(expStd, pearsonCorr)
covEE = Q3.generateCov(expStd, expCorr)
covs = [covPP, covPE, covEP, covEE]


for i in range(len(covs)):
    executionTime.append([])
    norms.append([])
    startTime = time.time()
    root = Q2.CholeskyPSD(covs[i])
    newCov = Q3.simulateCov(root, numOfDraws)
    executionTime[i].append(time.time() - startTime)
    norms[i].append(Q2.frobeniusNorm(newCov - covs[i]))
    for explainedPower in explainedPowers:
        startTime = time.time()
        root = Q3.findPCARoot(covs[i], explainedPower)
        newCov = Q3.simulateCov(root, numOfDraws)
        executionTime[i].append(time.time() - startTime)
        norms[i].append(Q2.frobeniusNorm(newCov - covs[i]))

executionTime = pd.DataFrame(executionTime)
norms = pd.DataFrame(norms)

executionTime.columns = [['directSimulation', 'PCA100%', 'PCA75%', 'PCA50%']]
executionTime.index = [['PCorr-PStd', 'PCorr-EStd', 'ECorr-PStd', 'ECorr-EStd']]
norms.columns = [['directSimulation', 'PCA100%', 'PCA75%', 'PCA50%']]
norms.index = [['PCorr-PStd', 'PCorr-EStd', 'ECorr-PStd', 'ECorr-EStd']]


print(executionTime)
print(norms)






