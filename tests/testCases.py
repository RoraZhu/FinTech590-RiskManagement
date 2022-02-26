import pandas as pd
import numpy as np
from scipy.stats import norm, t
from os.path import dirname, abspath
import RiskManagementPackage.VaR as VaR
import RiskManagementPackage.ES as ES
import RiskManagementPackage.MLE as mle
import RiskManagementPackage.Simulation as sim
import RiskManagementPackage.Factorization as fac
import RiskManagementPackage.CovarianceEstimation as covEst
import RiskManagementPackage.NonPSDFixed as psdFix


if __name__ == "__main__":
    # test VaR, ES and MLE
    path = dirname(dirname(abspath(__file__)))
    data = pd.read_csv(path + '/Week05/data/problem1.csv')
    data = np.array(data['x'].tolist())

    alpha = 0.05
    lam = 0.97
    size = 10000
    VaR1 = VaR.VaR_raw(data, alpha)
    VaR2 = VaR.VaR_distribution(norm, alpha, loc=data.mean(), scale=data.std())
    VaR3 = VaR.EWNormalVaR(alpha, lam, data)
    VaR4 = VaR.historicVaR(data, alpha)
    VaR5 = VaR.mleNormalVaR(alpha, data)
    VaR6 = VaR.mleTVaR(alpha, data)

    ES1 = ES.ES_raw(data, alpha)
    mean, sigma = mle.mleNormal(data)
    ES2 = ES.ES_distribution(norm, alpha, size=size, loc=mean, scale=sigma)
    df, mean, sigma = mle.mleT(data)
    ES3 = ES.ES_distribution(t, alpha, size=size, df=df, loc=mean, scale=sigma)

    print("VaR raw: ", VaR1)
    print("VaR with normal distribution: ", VaR2)
    print("VaR with EW sigma and normal distribution: ", VaR3)
    print("VaR with historical quantile: ", VaR4)
    print("VaR with mle normal: ", VaR5)
    print("VaR with mle t: ", VaR6)
    print("ES raw: ", ES1)
    print("ES with normal distribution: ", ES2)
    print("ES with t distribution: ", ES3)

    # test simulation, nonPSDFixed and covariance estimation
    numOfDraws = 10000

    path = dirname(dirname(abspath(__file__)))
    data = pd.read_csv(path + '/Week03/data/DailyReturn.csv')
    data = data.iloc[:, 1:]

    copulaSimulation = sim.copulaSimulation(data, numOfDraws)
    normalSimulation = sim.multiNormalSimulation(len(data), len(data.columns))

    data = data.T
    cov = pd.DataFrame(np.cov(data))
    corr = covEst.covToCorr(cov)
    nearPSD = psdFix.nearPSD(corr)
    nearestPSD = psdFix.nearestPSD(corr)
    root = fac.Cholesky(cov)
    print("correlation: ", cov)
    print("nearPSD: ", nearPSD)
    print("nearestPSD: ", nearestPSD)
    print("Cholesky: ", root)

    dataSimulation = sim.dataSimulation(root, numOfDraws)

    print("copula simulation: ", copulaSimulation)
    print("normal simulation: ", normalSimulation)
    print("data simulation: ", dataSimulation)

    # test covariance estimation
    print("exponentially weighted matrix: ", covEst.getEWCovMatrix(cov, lam))
    print("std of exponentially weighted matrix: ", covEst.ewCovToStd(cov, lam))
    print("correlation matrix of exponentially weighted matrix", covEst.ewCovToCorr(cov, lam))