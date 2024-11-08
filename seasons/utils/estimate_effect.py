import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from..utils import remove_trend

def compute_seasonal_effects(
    data: np.ndarray,
    seasons: np.ndarray,
    already_detrended: bool,
    trend = None,
    detrended = None,
    seasonality_type: str = 'auto',
    alpha: float = 0.05,
    display_plot: bool = False,
    return_effects: bool = True,
    use_linear_reg: bool = False,
    normalize: bool = False,
) -> dict:
    """
    Compute seasonal effects of a time series.

    Args:
    - data (np.ndarray): Input time series.
    - seasons (np.ndarray): Array of seasonal periods.
    - seasonality_type (str): Type of seasonality ('additive' or 'multiplicative' or 'auto'). Defaults to 'auto'.
    - repeat_seasons (bool, optional): Whether to repeat seasonal effects to match series length. Defaults to False.
    - display_plot (bool, optional): Whether to display the plot. Defaults to False.
    - return_effects (bool, optional): Whether to return a dict of the seasons and their effects
    """

    # Sort seasons
    seasons = np.sort(seasons)

    # Detrend
    if not already_detrended:
        trend, detrended, seasonality_type = remove_trend(data=data, seasonality_type=seasonality_type, use_linear_reg=use_linear_reg)
    N = len(detrended)
    
    # estimating effects
    c, seasons_effects = _confidence_interval(data=detrended, seasonal_periods=seasons, normalize=normalize, seasonality_type=seasonality_type, alpha=alpha)
    if normalize:
        if seasonality_type == 'additive':
            trend -= abs(c)
            detrended = data - trend
        else:
            trend *= c
            detrended = data / trend

    
    if display_plot:
        _plot_seasonal_components(
            seasonal_components=seasons_effects,
            data=data,
            trend=trend,
            detrended=detrended,
            seasonality_type=seasonality_type
        )

    if return_effects:
        return seasons_effects
    

def _plot_seasonal_components(
    seasonal_components: dict,
    data: np.typing.ArrayLike, 
    trend: np.typing.ArrayLike, 
    detrended: np.typing.ArrayLike, 
    seasonality_type: str
) -> None:
    """
    Plot seasonal components of a time series.

    Args:
    - seasonal_components: Dictionary with seasonal effects
    - data: Original time series data
    - detrended: Detrended time series data
    - seasonality_type: Type of seasonality (additive, multiplicative)

    Returns:
    - None
    """
    N_FIG = len(seasonal_components) + 4
    fig, ax = plt.subplots(nrows=N_FIG, ncols=1, figsize=(16, 6*N_FIG))

    ax[0].plot(range(1, len(data) + 1), data, label="Original series")
    ax[0].plot(range(1, len(trend) + 1), trend, label="Estimated trend")
    ax[0].legend()
    ax[1].plot(range(1, len(detrended) + 1), detrended, label=f"Detrended series, Seas. type={seasonality_type}")
    ax[1].legend()

    #seasonal_sums_lengths = sum([len(component["effect"]) for season, component in seasonal_components.items()])
    if seasonality_type == 'additive':
        total_seasonal_effect = np.zeros(len(data))
    else: 
        total_seasonal_effect = np.ones(len(data))

    for i, (season, component) in enumerate(seasonal_components.items()):
        effects = component["effect"]
        effects_to_add = _repeat_array_until_length(effects, len(total_seasonal_effect))
        if seasonality_type == 'additive':
            total_seasonal_effect += effects_to_add
        else: 
            total_seasonal_effect *= effects_to_add
        conf_intervals = component["confidence_interval"]        
        # Plot effects
        ax[i+4].plot(range(1, len(effects) + 1), effects, marker='o', label=f"Effect ({season})")
        ax[i+4].fill_between(
            range(1, len(effects) + 1),
            [interval[0] for interval in conf_intervals],
            [interval[1] for interval in conf_intervals],
            color="grey",
            alpha=0.3,
            label="Confidence Bound"
        )
        ax[i+4].legend()

    ax[2].plot(range(1, len(total_seasonal_effect) + 1), total_seasonal_effect, label=f"Total Seasonal Effect")
    ax[2].legend()
    #total_seasonal_effect_to_add = _repeat_array_until_length(total_seasonal_effect, len(detrended))
    if seasonality_type == 'additive':
        residuals = detrended - total_seasonal_effect
    else:
        residuals = detrended / total_seasonal_effect
    ax[3].plot(range(1, len(residuals) + 1), residuals, label=f"Residuals", marker='o')
    ax[3].legend()

    # Render plot
    plt.tight_layout()
    plt.suptitle("Seasonal Components")
    plt.show()

    return None


def _confidence_interval(
    data: np.ndarray,
    seasonal_periods: list,
    normalize: bool = False,
    seasonality_type: str = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compute confidence intervals for seasonal effects.

    Args:
    - data: Detrended time series data
    - seasonal_periods: List of seasonal periods
    - normalize: Whether to normalize the seasonal effects
    - seasonality_type: Type of seasonality (additive, multiplicative)
    - alpha: Confidence level

    Returns:
    - Dictionary with seasonal components and confidence intervals
    """
    X = _generate_design_matrix(len(data), seasonal_periods=seasonal_periods)
    #X = add_constant(X)
    model = OLS(data, X)
    model = model.fit()
    coefficients = model.params
    confidence_int = model.conf_int(alpha)
    # normalize
    min_coefficients = None
    if normalize:
        min_coefficients = np.min(coefficients)
        if seasonality_type == 'additive':
            coefficients += np.abs(min_coefficients)
            confidence_int = np.array([[lower + np.abs(min_coefficients), upper + np.abs(min_coefficients)] for (lower, upper) in confidence_int])
        elif seasonality_type == 'multiplicative':
            if min_coefficients != 0:
                coefficients *= (1/min_coefficients)
                confidence_int = np.array([[lower * (1/min_coefficients), upper * (1/min_coefficients)] for (lower, upper) in confidence_int])

    # Initialize output dictionary
    seasonal_components = {}
    
    # Iterate over seasonal periods and coefficients
    coeff_index = 0
    for period, period_name in enumerate(seasonal_periods, start=1):
        period_coeffs = coefficients[coeff_index:coeff_index + period_name]
        period_conf_int = confidence_int[coeff_index:coeff_index + period_name]
        
        # Store results in dictionary
        seasonal_components[f"s{period_name}"] = {
            "effect": period_coeffs.tolist(),
            "confidence_interval": period_conf_int.tolist()
        }
        
        # Update coefficient index
        coeff_index += period_name
    return min_coefficients, seasonal_components
    

def _generate_design_matrix(N, seasonal_periods):
    """
    Generate a design matrix for seasonal effect estimation.

    - N: Length of the time series
    - seasonal_periods: List of seasonal periods
    - kwargs: Reserved for future expansion

    Returns:
    - Design matrix as a dictionary
    """
    K = len(seasonal_periods)
    num_seasonal_effects = sum(seasonal_periods)
    A = np.zeros((N, num_seasonal_effects))
    
    effect_index = 0
    for k, xk in enumerate(seasonal_periods):
        for i in range(xk):
            for t in range(N):
                if t % xk == i:
                    A[t, effect_index + i] = 1
        effect_index += xk
    
    return A


def _repeat_array_until_length(arr, desired_length):
    """
    Repeat the input array until it reaches the desired length.

    Parameters:
    - arr (list or np.ndarray): Input array to be repeated.
    - desired_length (int): Desired length of the output array.

    Returns:
    - np.ndarray: The repeated array, truncated to the desired length.
    """
    arr = np.asarray(arr)  # Ensure input is a NumPy array
    repetitions = int(np.ceil(desired_length / len(arr)))  # Calculate repetitions needed
    repeated_arr = np.tile(arr, repetitions)  # Repeat the array
    return repeated_arr[:desired_length]  # Truncate to the desired length
