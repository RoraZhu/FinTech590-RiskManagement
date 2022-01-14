# Project Response

## Problem 1

#### The conditional distribution of the Multivariate Normal to the OLS equations are the same.

- <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午10.15.01.png" style="zoom:50%;" />

#### Reason

- OLS is to estimate Y given X, which has the same meaning of finding the conditional distribution of Y given X. If the covariance between X and Y is not 0, the mean of the conditional distribution of Y has to adjust according to the covariance between X and Y.
- <img src="/Users/rora/PycharmProjects/FinTech590-RiskManagement/Week02/images/截屏2022-01-13 下午10.28.38.png" style="zoom:25%;" />

- This graph vividly shows the reason why the two are the same. The graph shows that the margin distribution of X and Y are both normal. Also, X and Y are positively correlated. the fitted Y given X=x0 is exactly the mean of Y's normal distribution given X=x0. The conditional mean adjusts when X changes.

## Problem 2

### OLS

#### Distribution of the error vector using OLS

<img src="/Users/rora/PycharmProjects/FinTech590-RiskManagement/Week02/images/截屏2022-01-13 下午10.28.08.png" style="zoom:50%;" />

#### How well does it fit the assumption of normally distributed errors?

- To know how well it fits the assumption of normally distributed errors, I use qqplot and SW test to estimate its normality. qqplot shows the distribution of the data against the expected normal distribution. For normally distributed data, observations should lie approximately on a straight line. Shapiro-Wilk test is a test of normality.

  - qqplot

    <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午10.34.13.png" alt="截屏2022-01-13 下午10.34.13" style="zoom:50%;" />

  - SW test

    <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午10.34.53.png" style="zoom:50%;" />

- As seen in the pictures, the histogram of the error vector is skewed, the qqplot is not straight, and the SW test reject the hypothesis that the Sample is an Gaussian.
- Therefore, it does not fit the assumption of normally distributed errors.

### MLE

#### MLE with normal assumption and t assumption

<img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午10.45.18.png" alt="截屏2022-01-13 下午10.45.18" style="zoom:50%;" />

- The MLE using the assumption of a T distribution of the errors is the best fit.

### Comparing parameters

#### What are the fitted parameters of each and how do they compare?

- The fitted parameters of OLS are the same as the fitted parameters of MLE with normal assumption. 
- The fitted parameters of MLE with normal assumption are different from those of the MLE with t assumption.

- <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.09.58.png" alt="截屏2022-01-13 下午11.09.58" style="zoom:50%;" />

#### What does this tell us about the breaking of the normality assumption in regards to expected values in this case?

- The breaking of the normality assumption will influence parameter estimation

## Problem 3

### Simulate AR(1) through AR(3) and MA(1) through MA(3) processes

- ```python
  # AR1
  ar = np.array([1, -0.9])
  ma = np.array([1])
  AR = ArmaProcess(ar, ma)
  simulated_ar1 = AR.generate_sample(nsample=1000)
  plt.plot(simulated_ar1)
  plt.show()
  ```

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.22.17.png" alt="截屏2022-01-13 下午11.22.17" style="zoom:50%;" />

  ```python
  # AR2
  ar = np.array([1, -0.9, 0.3])
  ma = np.array([1])
  AR = ArmaProcess(ar, ma)
  simulated_ar2 = AR.generate_sample(nsample=1000)
  plt.plot(simulated_ar2)
  plt.show()
  ```

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.22.37.png" alt="截屏2022-01-13 下午11.22.37" style="zoom:50%;" />

  ```python
  # AR3
  ar = np.array([1, -0.9, 0.3, 0.2])
  ma = np.array([1])
  AR = ArmaProcess(ar, ma)
  simulated_ar3 = AR.generate_sample(nsample=1000)
  plt.plot(simulated_ar3)
  plt.show()
  ```

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.22.59.png" alt="截屏2022-01-13 下午11.22.59" style="zoom:50%;" />

  ```python
  # MA1
  ar1 = np.array([1])
  ma1 = np.array([1, -0.9])
  MA = ArmaProcess(ar1, ma1)
  simulated_ma1 = MA.generate_sample(nsample=1000)
  plt.plot(simulated_ma1)
  plt.show()
  ```

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.23.17.png" alt="截屏2022-01-13 下午11.23.17" style="zoom:50%;" />

  ```python
  # MA2
  ma = np.array([1, -0.9, 0.3])
  ar = np.array([1])
  MA = ArmaProcess(ar, ma)
  simulated_ma2 = MA.generate_sample(nsample=1000)
  plt.plot(simulated_ma2)
  plt.show()
  ```

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.23.39.png" alt="截屏2022-01-13 下午11.23.39" style="zoom:50%;" />

  ```python
  # MA3
  ma = np.array([1, -0.9, 0.3, 0.2])
  ar = np.array([1])
  MA = ArmaProcess(ar, ma)
  simulated_ma3 = MA.generate_sample(nsample=1000)
  plt.plot(simulated_ma3)
  plt.show()
  ```

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.23.58.png" alt="截屏2022-01-13 下午11.23.58" style="zoom:50%;" />

### ACF and PACF

#### ACF

- AR1

  

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.26.09.png" alt="截屏2022-01-13 下午11.26.09" style="zoom:50%;" />

- AR2

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.26.42.png" alt="截屏2022-01-13 下午11.26.42" style="zoom:50%;" />

- AR3

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.26.59.png" alt="截屏2022-01-13 下午11.26.59" style="zoom:50%;" />

- MA1

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.27.18.png" alt="截屏2022-01-13 下午11.27.18" style="zoom:50%;" />

- MA2

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.27.39.png" alt="截屏2022-01-13 下午11.27.39" style="zoom:50%;" />

- MA3

  <img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.27.54.png" alt="截屏2022-01-13 下午11.27.54" style="zoom:50%;" />

#### PACF

- AR1

<img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.29.48.png" alt="截屏2022-01-13 下午11.29.48" style="zoom:50%;" />

- AR2

<img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.30.07.png" alt="截屏2022-01-13 下午11.30.07" style="zoom:50%;" />

- AR3

<img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.30.25.png" alt="截屏2022-01-13 下午11.30.25" style="zoom:50%;" />

- MA1

<img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.31.48.png" alt="截屏2022-01-13 下午11.31.48" style="zoom:50%;" />

- MA2

<img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.32.07.png" alt="截屏2022-01-13 下午11.32.07" style="zoom:50%;" />

- MA3

<img src="/Users/rora/Library/Application Support/typora-user-images/截屏2022-01-13 下午11.32.24.png" alt="截屏2022-01-13 下午11.32.24" style="zoom:50%;" />

#### Identify the type and order of each process

- If a process is an AR process, its autocorrelation will decrease, or oscillate to decrease slowly, and the number of lags that are significantly differ from 0 in the partial autocorrelation indicates the order of this AR process.
- If a process is an MA process, its partial autocorrelation will decrease, or oscillate to decrease slowly, and the number of lags that are significantly differ from 0 in the autocorrelation indicates the order of this MA process.