import numpy as np

def generate_series(
    length: int,
    trend: str = None,
    trend_coefficient: float = 1.0,
    seasonality: list = None,
    seasonality_type: str = 'additive',
    error_distribution: str = 'normal',
    error_scale: float = 1.0
) -> np.typing.ArrayLike:
    """
    Generate a time series with trend, seasonality, and error components.

    Args:
    - length (int): Length of the time series.
    - trend (str, optional): Type of trend ('linear', 'quadratic', or 'none'). Defaults to 'linear'.
    - trend_coefficient (float, optional): Coefficient for the trend. Defaults to 1.0.
    - seasonality (list, optional): List of seasonality components, where each component is a list of values.
        For additive seasonality, values are added to the trend.
        For multiplicative seasonality, values are multiplied with the trend.
        Defaults to None.
    - seasonality_type (str, optional): Type of seasonality ('additive' or 'ultiplicative'). Defaults to 'additive'.
    - error_distribution (str, optional): Distribution of the error term ('normal' or 'uniform'). Defaults to 'normal'.
    - error_scale (float, optional): Scale of the error term. Defaults to 1.0.

    Returns:
    - pd.Series: Generated time series.
    """

    # Generate trend component
    if trend == 'linear':
        trend_component = trend_coefficient * np.arange(length)
    elif trend == 'quadratic':
        trend_component = trend_coefficient * np.arange(length) ** 2
    elif trend == None:
        trend_component = np.zeros(length)
    else:
        raise ValueError("Invalid trend type")

    # Generate seasonality component(s)
    if seasonality is not None:
        seasonality_components = []
        for seasonal_values in seasonality:
            # Broadcast seasonality values to the length of the series
            seasonal_component = np.tile(seasonal_values, int(np.ceil(length / len(seasonal_values))))[:length]
            seasonality_components.append(seasonal_component)
    else:
        seasonality_components = [np.ones(length)]

    # Combine seasonality components (if multiple)
    if seasonality_type == 'additive':
        combined_seasonality = np.sum(seasonality_components, axis=0)
    elif seasonality_type == 'multiplicative':
        combined_seasonality = np.prod(seasonality_components, axis=0)
    else:
        raise ValueError("Invalid seasonality type")

    # Generate error component
    if error_distribution == 'normal':
        error_component = np.random.normal(scale=error_scale, size=length)
    elif error_distribution == 'uniform':
        error_component = np.random.uniform(low=-error_scale, high=error_scale, size=length)
    else:
        raise ValueError("Invalid error distribution")

    # Combine trend, seasonality, and error components
    if seasonality_type == 'additive':
        time_series = trend_component + combined_seasonality + error_component
    elif seasonality_type == 'multiplicative':
        time_series = trend_component * combined_seasonality * (1 + error_component)

    return time_series