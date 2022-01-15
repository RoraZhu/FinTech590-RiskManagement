import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(12345)

# AR1
ar = np.array([1, -0.9])
ma = np.array([1])
AR = ArmaProcess(ar, ma)
simulated_ar1 = AR.generate_sample(nsample=1000)
# plt.plot(simulated_ar1)
# plt.show()

# AR2
ar = np.array([1, -0.9, 0.3])
ma = np.array([1])
AR = ArmaProcess(ar, ma)
simulated_ar2 = AR.generate_sample(nsample=1000)
# plt.plot(simulated_ar2)
# plt.show()

# AR3
ar = np.array([1, -0.9, 0.3, 0.2])
ma = np.array([1])
AR = ArmaProcess(ar, ma)
simulated_ar3 = AR.generate_sample(nsample=1000)
# plt.plot(simulated_ar3)
# plt.show()

# MA1
ar1 = np.array([1])
ma1 = np.array([1, -0.9])
MA = ArmaProcess(ar1, ma1)
simulated_ma1 = MA.generate_sample(nsample=1000)
# plt.plot(simulated_ma1)
# plt.show()

# MA2
ma = np.array([1, -0.9, 0.3])
ar = np.array([1])
MA = ArmaProcess(ar, ma)
simulated_ma2 = MA.generate_sample(nsample=1000)
# plt.plot(simulated_ma2)
# plt.show()

# MA3
ma = np.array([1, -0.9, 0.3, 0.2])
ar = np.array([1])
MA = ArmaProcess(ar, ma)
simulated_ma3 = MA.generate_sample(nsample=1000)
# plt.plot(simulated_ma3)
# plt.show()

# plot_acf(simulated_ar1, lags=20)
# plot_acf(simulated_ar2, lags=20)
# plot_acf(simulated_ar3, lags=20)
# plot_acf(simulated_ma1, lags=20)
# plot_acf(simulated_ma2, lags=20)
# plot_acf(simulated_ma3, lags=20)
# plt.show()
#
# plot_pacf(simulated_ar1, lags=20)
# plot_pacf(simulated_ar2, lags=20)
# plot_pacf(simulated_ar3, lags=20)
# plot_pacf(simulated_ma1, lags=20)
# plot_pacf(simulated_ma2, lags=20)
# plot_pacf(simulated_ma3, lags=20)
# plt.show()

fig, axs = plt.subplots(3, 1)
axs[0].plot(simulated_ar1)
axs[0].set_title("AR1")
axs[1].plot(simulated_ar2)
axs[1].set_title("AR2")
axs[2].plot(simulated_ar2)
axs[2].set_title("AR3")
plt.show()

fig, axs = plt.subplots(3, 1)
axs[0].plot(simulated_ma1)
axs[0].set_title("MA1")
axs[1].plot(simulated_ma2)
axs[1].set_title("MA2")
axs[2].plot(simulated_ma2)
axs[2].set_title("MA3")
plt.show()

fig, ax = plt.subplots(6, 2, figsize=(10, 30))

plot_acf(simulated_ar1, lags=20, ax=ax[0][0])
plot_pacf(simulated_ar1, lags=20, ax=ax[0][1])
ax[0][0].set_title("AR1 Autocorrelation")

plot_acf(simulated_ar2, lags=20, ax=ax[1][0])
plot_pacf(simulated_ar2, lags=20, ax=ax[1][1])
ax[1][0].set_title("AR2 Autocorrelation")

plot_acf(simulated_ar3, lags=20, ax=ax[2][0])
plot_pacf(simulated_ar3, lags=20, ax=ax[2][1])
ax[2][0].set_title("AR3 Autocorrelation")

plot_acf(simulated_ma1, lags=20, ax=ax[3][0])
plot_pacf(simulated_ma1, lags=20, ax=ax[3][1])
ax[3][0].set_title("MA1 Autocorrelation")

plot_acf(simulated_ma2, lags=20, ax=ax[4][0])
plot_pacf(simulated_ma2, lags=20, ax=ax[4][1])
ax[4][0].set_title("MA2 Autocorrelation")

plot_acf(simulated_ma3, lags=20, ax=ax[5][0])
plot_pacf(simulated_ma3, lags=20, ax=ax[5][1])
ax[5][0].set_title("MA3 Autocorrelation")
plt.show()
plt.savefig('acf_pacf.png')