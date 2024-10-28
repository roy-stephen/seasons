import numpy as np
from statsmodels.tsa.stattools import adfuller

def stationarize(
    data: np.typing.ArrayLike, 
    alpha: float = 0.05
) -> tuple[np.typing.ArrayLike, int]:
    """
    Stationarize a time series by applying differencing until the Augmented Dickey-Fuller (ADF) test indicates stationarity.

    Args:
    - data (np.typing.ArrayLike): Input time series data.
    - alpha (float, optional): Significance level for the ADF test. Defaults to 0.05.

    Returns:
    - tuple[np.typing.ArrayLike, int]: A tuple containing the stationarized data and the order of integration (i.e., the number of times differencing was applied).

    Notes:
    - If the input data is constant, it is considered already stationarized, and the original data is returned with an integration order of 0.
    """

    # Initialize variables
    integration_order = 0

    # Check if data is constant
    if np.unique(data).shape[0] == 1:
        return data, integration_order

    # Perform ADF test and apply differencing as needed
    adf_result = adfuller(data)
    p_value_adf = adf_result[1]  # p-value from ADF test

    while p_value_adf >= alpha:
        data = np.diff(data)
        integration_order += 1
        adf_result = adfuller(data)
        p_value_adf = adf_result[1]  # Update p-value from ADF test

    return data, integration_order