import numpy as np

def generate_series(
    length: int,
    seasonality_periods: list[int] = None,
    seasonality_amplitudes: list[float] = None,
    seasonality_types: list[str] = None,  # 'additive' or 'multiplicative'
    trend: str = None,  # 'linear', 'quadratic', or None
    trend_coefficient: float = 1.0,
    noise: bool = False,  # True or False
    noise_stddev: float = 1.0
) -> np.ndarray:
    """
    Generate a synthetic time series with desired characteristics.

    Args:
    - length (int): Desired length of the time series.
    - seasonality_periods (list[int], optional): List of seasonality periods. Defaults to None.
    - seasonality_amplitudes (list[float], optional): List of amplitudes for each seasonality period. Defaults to None.
    - seasonality_types (list[str], optional): List of seasonality types ('additive' or 'multiplicative') for each period. Defaults to None.
    - trend (str, optional): Type of trend ('linear', 'quadratic', or None). Defaults to None.
    - trend_coefficient (float, optional): Coefficient for the trend. Defaults to 1.0.
    - noise (bool, optional): To add noise (True or False). Defaults to False.
    - noise_stddev (float, optional): Standard deviation for Gaussian noise. Defaults to 1.0.

    Returns:
    - np.ndarray: Generated time series.
    """

    # Initialize the time series with zeros
    time_series = np.zeros(length)

    # Add seasonality
    if seasonality_periods is not None:
        if seasonality_amplitudes is None:
            seasonality_amplitudes = [1.0] * len(seasonality_periods)
        if seasonality_types is None:
            seasonality_types = ['additive'] * len(seasonality_periods)

        for period, amplitude, s_type in zip(seasonality_periods, seasonality_amplitudes, seasonality_types):
            seasonal_component = amplitude * np.sin(2 * np.pi * np.arange(length) / period)
            if s_type == 'additive':
                time_series += seasonal_component
            elif s_type == 'multiplicative':
                time_series *= (1 + seasonal_component)

    # Add trend
    if trend == 'linear':
        time_series += trend_coefficient * np.arange(length)
    elif trend == 'quadratic':
        time_series += trend_coefficient * (np.arange(length) ** 2)

    # Add noise
    if noise:
        time_series += np.random.normal(scale=noise_stddev, size=length)

    return time_series