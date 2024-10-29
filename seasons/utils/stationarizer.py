import numpy as np
from scipy.stats import bartlett
from statsmodels.tsa.stattools import adfuller


def stationarize(data, alpha=0.05, seasonality_type='auto'):
    """
    Stationarize a time series based on the specified seasonality type.

    Args:
    - data (np.ndarray): Input time series.
    - alpha (float, optional): Significance level. Defaults to 0.05.
    - seasonality_type (str, optional): Type of seasonality. Defaults to 'auto'.

    Returns:
    - np.ndarray: Stationarized time series.
    - int: Order of integration.
    """
    if seasonality_type == 'additive':
        # Regular differencing until stationary
        stationarized, integration_order = _differencing_until_stationary(data, alpha)
    elif seasonality_type == 'multiplicative':
        # Compute percentage changes with special handling
        stationarized = _safe_percentage_change(data)
        integration_order = 1  # Assuming one order of integration for multiplicative seasonality
    elif seasonality_type == 'auto':
        # First perform differencing until stationary
        stationarized, integration_order = _differencing_until_stationary(data, alpha)
        # Check if variance is constant
        if _is_variance_constant(stationarized):
            # Proceed with additive seasonality
            print("Seasonality is likely additive.")
            seasonality_type = 'additive'
            pass
        else:
            # Apply steps for multiplicative seasonality
            print("Seasonality is likely multiplicative.")
            seasonality_type = 'multiplicative'
            stationarized = _safe_percentage_change(data)
            integration_order = 1  # Assuming one order of integration for multiplicative seasonality
    else:
        raise ValueError("Invalid seasonality_type. Choose from 'additive', 'multiplicative', or 'auto'.")

    return stationarized, integration_order, seasonality_type


def _differencing_until_stationary(
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

    # Convert input to numpy array if it's a list
    data = np.array(data)

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


def _safe_percentage_change(values):
    """
    Compute percentage changes while handling infinite and near-zero values.

    Args:
    - values (np.ndarray): Input values.
    Returns:
    - np.ndarray: Percentage changes with special handling for infinite and near-zero values.
    """
    # Compute percentage changes handling for negative values
    percentage_changes = np.diff(values) / np.abs(values[:-1])

    existing_max = _get_max_excluding_indices(percentage_changes, []) # get max values without nan or inf

    # Handle infinite values 
    infinite_mask = np.isinf(percentage_changes)
    if np.any(infinite_mask):
        # Return current max for inf for infinite values
        percentage_changes[infinite_mask] = existing_max

    # Handle abnormaly large values (where input was really close to 0)
    oultiers = _detect_outliers(percentage_changes)
    existing_max = _get_max_excluding_indices(percentage_changes, oultiers)
    if any(oultiers):
        percentage_changes[oultiers] = existing_max

    return percentage_changes

def _get_max_excluding_indices(array, excluded_indices):
    # Create a boolean mask to exclude the specified indices
    mask = ~np.isin(range(len(array)), excluded_indices)
    
    # Filter the array using the mask
    filtered_array = array[mask]
    filtered_array = filtered_array[np.isfinite(filtered_array)]
    
    # Get the maximum of the filtered array
    max_value = np.max(filtered_array)
    
    return max_value

def _detect_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (3 * iqr)
    upper_bound = q3 + (3 * iqr)
    return np.where((data < lower_bound) | (data > upper_bound))[0]

def _is_variance_constant(data, alpha=0.05, num_segments=2):
    """
    Determine if the variance of a time series is constant using Levene's test.

    Args:
    - data (np.ndarray): Input time series.
    - alpha (float, optional): Significance level. Defaults to 0.05.
    - num_segments (int, optional): Number of segments to divide the data. Defaults to 2.

    Returns:
    - bool: True if the variance is constant, False otherwise.
    """
    # Split the data into multiple segments
    segment_size = len(data) // num_segments
    segments = [data[i*segment_size:(i+1)*segment_size] for i in range(num_segments)]

    # Perform Levene's test
    _, p_value = bartlett(*segments)

    # Check if the p-value is greater than the significance level
        # Fail to reject the null hypothesis (variance is constant)
    return p_value > alpha