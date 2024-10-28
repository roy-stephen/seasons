import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from..utils import stationarize, compute_cartesian_products  # Import stationarize from utils module

def brute_force_seasonality(
    data: np.typing.ArrayLike, 
    alpha: float = 0.05, 
    min_seasonality: int = 2,
    apply_cartesian: bool = False
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

    # Stationarize data
    ts, _ = stationarize(data=data)

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

        # Check if seasonality is significant
        if p_value < alpha:
            best_seasonality.append(seasonality)

    return compute_cartesian_products(best_seasonality) if apply_cartesian else best_seasonality 