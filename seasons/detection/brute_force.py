import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from..utils import stationarize  # Import stationarize from utils module

def brute_force_seasonality(
    data: np.typing.ArrayLike, 
    alpha: float = 0.05, 
    min_seasonality: int = 2
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
    max_seasonality = len(ts) // 2

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

    return best_seasonality


# def _test_seasonality(seasonality, ts, alpha):
#     """
#     Helper function to test a single seasonality period.

#     Args:
#     - seasonality (int): Seasonality period to test.
#     - ts (np.ndarray): Stationarized time series.
#     - alpha (float): Significance level for the ANOVA test.

#     Returns:
#     - int or None: Seasonality period if significant, otherwise None.
#     """
#     reshaped_data = ts[:len(ts) // seasonality * seasonality].reshape(-1, seasonality)
#     _, p_value = stats.f_oneway(*reshaped_data.T)
#     return seasonality if p_value < alpha else None

# def brute_force_seasonality_parallel(
#     data: np.typing.ArrayLike, 
#     alpha: float = 0.05, 
#     n_jobs: int = -1
# ) -> list[int]:
#     """
#     Parallelized brute-force seasonality detection using one-way ANOVA.

#     Args:
#     - data (np.typing.ArrayLike): Input time series data.
#     - alpha (float, optional): Significance level for the ANOVA test. Defaults to 0.05.
#     - n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (using all available cores).

#     Returns:
#     - list[int]: List of detected seasonality periods.
#     """
#     ts, _ = stationarize(data)
#     max_seasonality = len(ts) // 2
#     seasonalities = range(2, max_seasonality + 1)

#     results = Parallel(n_jobs=n_jobs)(delayed(_test_seasonality)(seas, ts, alpha) for seas in seasonalities)
#     return [seas for seas in results if seas is not None]