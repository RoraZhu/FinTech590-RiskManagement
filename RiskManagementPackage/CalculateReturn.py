import numpy as np


def return_calculate(prices, method="DISCRETE", dateColumn="date"):
    if dateColumn not in prices.columns:
        print("dateColumn: ", dateColumn, " not in DataFrame: ", prices)
        return
    rets = prices.copy()
    if method.upper() == "DISCRETE":
        for column in prices.columns[1:]:
            rets[column] = prices[column] / prices[column].shift() - 1
    elif method.upper() == "LOG":
        for column in prices.columns[1:]:
            rets[column] = np.log(prices[column] / prices[column].shift())
    else:
        print("method: ", method, " must be DISCRETE or LOG")
    return rets.iloc[1:, :]