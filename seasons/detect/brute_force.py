import numpy as np
from scipy import stats
from..utils import remove_trend, compute_cartesian_products, compute_seasonal_effects

def brute_force_seasonality(
    data: np.typing.ArrayLike, 
    alpha: float = 0.05, 
    min_seasonality: int = 2,
    seasonality_type: str = 'auto',
    apply_cartesian: bool = False,
    return_effects: bool = False,
    display_plot: bool = False,
    use_linear_reg: bool = False,
) -> list[int]:
    """
    Brute-force seasonality detector using one-way ANOVA.

    Args:
    - data (np.typing.ArrayLike): Input time series data.
    - alpha (float, optional): Significance level for the ANOVA test. Defaults to 0.05.
    - min_seasonality (int, optional): Minimum seasonality period to consider. Defaults to 2.

    Returns:
    - list[int]: List of detected seasonality periods.
    """

    # Detrend data
    trend, ts, seasonality_type = remove_trend(data=data, seasonality_type=seasonality_type, use_linear_reg=use_linear_reg)
    
    # Handle NA
    if seasonality_type == 'multiplicative':
        ts = np.nan_to_num(x=ts, nan=0)
        

    # Initialize variables
    best_seasonality = []
    max_seasonality = int(len(ts) // 3) # Set the maximum seasonality period to approximately 33% of the time series length
    # Rationale:
    # - Ensure the seasonality pattern has sufficient repetitions to be confirmed (at least 3 full cycles)
    # - This threshold helps prevent false positives by requiring a more robust seasonal signal

    # Loop through possible seasonality periods
    for seasonality in range(min_seasonality, max_seasonality + 1):
        # Skip if seasonality is a multiple of already found periods
        if any(seasonality % seas == 0 for seas in best_seasonality):
            continue

        # Reshape data to apply ANOVA
        reshaped_data = ts[:len(ts) // seasonality * seasonality].reshape(-1, seasonality)


        # Apply one-way ANOVA
        _, p_value = stats.f_oneway(*reshaped_data.T)
        # here by transposing i am comparing the first element of each seasonl period to the second, the third, ...
        # ideally i should perform the anova without rechaping and look for seasonalities where p-value >= thr.
        # however, low nimber of observations make this approach non bearable.
        # although this one is not the best one, it is hard to find data where this approach find a seasonal period that
        # is not somehow valid.

        # Check if seasonality is significant
        if p_value < alpha:
            best_seasonality.append(seasonality)

    if apply_cartesian:
        best_seasonality = compute_cartesian_products(best_seasonality, len(ts))
    else:
        pass

    if not return_effects: 
        return seasonality_type, best_seasonality

    if best_seasonality:
        d = compute_seasonal_effects(
            data=data, 
            already_detrended=True,
            trend=trend,
            detrended=ts,
            seasons=best_seasonality,
            alpha=alpha, 
            seasonality_type=seasonality_type,
            return_effects=True,
            display_plot=display_plot,
            use_linear_reg=use_linear_reg
        )
    else:
        d = {}

    return seasonality_type, d