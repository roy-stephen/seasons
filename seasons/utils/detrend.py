import numpy as np
from scipy.stats import bartlett, f_oneway
from statsmodels.api import add_constant, OLS
from PyEMD import EMD

def remove_trend(
    data: np.ndarray, 
    seasonality_type: str = 'auto', 
    use_linear_reg: bool = False
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Stationarize a time series by removing the trend based on the specified seasonality type.

    Args:
    - data (np.ndarray): Input time series.
    - seasonality_type (str, optional): Type of seasonality. Defaults to 'auto'.
        - 'additive': Assume additive seasonality.
        - 'multiplicative': Assume multiplicative seasonality.
        - 'auto': Automatically detect seasonality type.

    Returns:
    - trend (np.ndarray): Estimated trend of the input time series.
    - detrended (np.ndarray): Detrended time series.
    - seasonality_type (str): Detected or specified seasonality type.
    """
    trend = _estimate_and_validate_trend(data, use_linear_reg)
    
    if seasonality_type == 'additive':
        detrended = data - trend
    elif seasonality_type == 'multiplicative':
        detrended = data / trend
    elif seasonality_type == 'auto':
        # Assume additive initially and adjust if necessary
        detrended = data - trend  
        if _is_variance_constant(detrended):
            seasonality_type = 'additive'
        else:
            seasonality_type = 'multiplicative'
            detrended = data / trend
    else:
        raise ValueError("Invalid seasonality_type. Choose from 'additive', 'multiplicative', or 'auto'.")

    return trend, detrended, seasonality_type


def _estimate_and_validate_trend(
    data: np.ndarray, 
    use_linear_reg: bool
) -> np.ndarray:
    """
    Estimate the trend of a time series and validate its accuracy.

    Args:
    - data (np.ndarray): Input time series.
    - use_linear_reg (bool): Whether to use linear regression for trend estimation.

    Returns:
    - trend (np.ndarray): Estimated trend of the input time series.
    """
    trend = _estimate_trend(data, use_linear_reg)
    
    if not use_linear_reg:
        # Iteratively refine the trend estimation until no seasonality remains
        second_trend = _estimate_trend(trend, use_linear_reg)
        while _has_seasonality(trend - second_trend):
            trend = _estimate_trend(trend, use_linear_reg)
            second_trend = _estimate_trend(trend, use_linear_reg)
    
    return trend


def _estimate_trend(
    data: np.ndarray, 
    use_linear_reg: bool
) -> np.ndarray:
    """
    Estimate the trend of a time series using either enveloppe from Empirical Mode Decomposition (EMD) or linear regression.

    Args:
    - data (np.ndarray): Input time series.
    - use_linear_reg (bool): Whether to use linear regression for trend estimation.

    Returns:
    - trend (np.ndarray): Estimated trend of the input time series.
    """
    emd = EMD()
    emd_results = emd.emd(S=data)
    
    if (len(emd_results) == 1) or use_linear_reg:
        # Use linear regression for trend estimation
        X = np.arange(len(data))
        X = add_constant(X)
        model = OLS(data, X)
        fitted = model.fit()
        trend = fitted.fittedvalues
    else:
        # Use EMD with max-min spline for trend estimation
        max_spline = emd.extract_max_min_spline(T=np.arange(1, len(data)+1), S=data)[0]
        min_spline = emd.extract_max_min_spline(T=np.arange(1, len(data)+1), S=data)[1]
        trend = (max_spline + min_spline) / 2
    
    return trend


def _has_seasonality(
    data: np.ndarray
) -> bool:
    """
    Detect whether a time series exhibits seasonality.

    Args:
    - data (np.ndarray): Input time series.

    Returns:
    - bool: True if the time series has seasonality, False otherwise.
    """
    max_seasonality = int(len(data) // 3)
    best_seasonality = []
    
    # Check for seasonality across various periods
    for seasonality in range(2, max_seasonality + 1):
        if any(seasonality % seas == 0 for seas in best_seasonality):
            continue
        reshaped_data = data[:len(data) // seasonality * seasonality].reshape(-1, seasonality)
        _, p_value = f_oneway(*reshaped_data.T)
        
        # Significant difference across seasons indicates seasonality
        if p_value < 0.05:
            best_seasonality.append(seasonality)
    
    return bool(best_seasonality)


def _is_variance_constant(
    data: np.ndarray, 
    alpha: float = 0.05, 
    num_segments: int = 2
) -> bool:
    """
    Determine if the variance of a time series is constant using Bartlett's test.

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

    # Perform bartlett's test
    _, p_value = bartlett(*segments)

    # Check if the p-value is greater than the significance level
    # Fail to reject the null hypothesis (variance is constant)
    return p_value > alpha