import numpy as np
import matplotlib.pyplot as plt
# from fractions import Fraction
from scipy import stats
from..detect import brute_force_seasonality
from..utils import stationarize


def plot_seasonal_components(
        series: np.typing.ArrayLike,
        seasons: np.typing.ArrayLike = None,
        seasonality_type: str = 'auto',
        alpha: float = 0.05
):
    # TODO: fix this chunk
    # if not isinstance(series, np.typing.ArrayLike):
    #     raise ValueError("Invalid series type")
    # if seasons and not isinstance(seasons, np.typing.ArrayLike):
    #     raise ValueError("Invalid seasons type")
    
    # Check if all items of seasons are integer # TODO: allow for decimal seasons next
    if seasons: # if the user specify some seasonal periods
        if not all(isinstance(s, int) for s in seasons):
            print("Decimal seasonality periods are not currenytly supported. Will round to integer.")
            seasons = [int(s) for s in seasons]
    else: # if the user don't specify any seasons
        print("No seasons were specified.\nUsing bruteforce to estimate seasonal periods...")
        seasonality_type, seasons = brute_force_seasonality(series, alpha=alpha, apply_cartesian=False, seasonality_type=seasonality_type)
        print(f"Detected seasons: {seasons}.")
    
    # Stationarize
    stationarized, integration_order, seasonality_type = stationarize(data=series, alpha=0.05, seasonality_type=seasonality_type)
    N = len(stationarized)
    
    # Initialize plot
    N_FIG = len(seasons) + 4 # 4 spaces for the orignal series, the stationary series, the sum of the seasonal effects and the residuals
    fig, ax = plt.subplots(nrows=N_FIG, ncols=1, figsize=(16, 6*N_FIG))
    # Plot series
    ax[0].plot(series, label="Original series")
    ax[0].legend()
    ax[1].plot(stationarized, label=f"Stationary series, d={int(integration_order)}, Seas. type={seasonality_type}")
    ax[1].legend()

    # initialize total seasonal effect
    sum_seasonal = 0
    # Estimate seasonal average for each seasons
    for i, s in enumerate(seasons):
        # Reshape
        reshaped = _reshape_to_2d(stationarized, s)
        # Compute means for seasonal effect estimation and stats for confidence interval
        avg, lower_bound, upper_bound = _confidence_interval(data=reshaped)
        # Harmonize size
        avg = _repeat_array_until_length(avg, N)
        lower_bound = _repeat_array_until_length(lower_bound, N)
        upper_bound = _repeat_array_until_length(upper_bound, N)
        # Update total seasonal effect
        sum_seasonal += avg
        # Plot component
        ax[i+4].plot(avg, label=f"Seasonal Component s={s}, d={integration_order}")
        ax[i+4].fill_between(
            range(N),
            upper_bound,
            lower_bound,
            color="grey",
            alpha=0.3,
            label="Confidence Bound"
        )
        ax[i+4].legend()
    # Plot sum seasons
    ax[2].plot(sum_seasonal, label=f"Total Seasonal Effect, d={integration_order}")
    ax[2].legend()
    # Plot residuals
    residuals = stationarized - sum_seasonal
    ax[3].plot(residuals, label=f"Residuals, d={integration_order}")
    ax[3].legend()
    # Render plot
    plt.tight_layout()
    plt.suptitle("Seasonal Components")
    plt.show()

    return None


def _reshape_to_2d(array, rows):
    """
    Reshape a 1D array to a 2D shape with a specified number of rows, 
    automatically determining the minimum number of columns required to 
    ensure the output array has at least as many elements as the input array, 
    filling with np.nan if necessary, and filling in column-first order.

    Parameters:
    - array (np.ndarray): The input 1D NumPy array.
    - rows (int): The desired number of rows for the output 2D array.

    Returns:
    - reshaped_array (np.ndarray): The reshaped 2D NumPy array.
    """
    # Calculate the minimum number of columns required
    cols = max(np.ceil(array.size / rows).astype(int), 1)  # Ensure cols is at least 1
    
    # Calculate the total number of elements in the new shape
    total_elements_needed = rows * cols
    
    # Extend the original array with np.nan if necessary
    if total_elements_needed > array.size:
        extended_array = np.append(array, np.full(total_elements_needed - array.size, np.nan))
    else:
        extended_array = array
    
    # Reshape the (possibly extended) array to the desired 2D shape in column-first order
    reshaped_array = extended_array.reshape(cols, rows, order='C').T  # Transpose to get rows first in output
    
    return reshaped_array

def _confidence_interval(
    data: np.ndarray,
    alpha: float = 0.05,
    return_means: bool = True
) -> tuple:
    """
    Compute the confidence intervals for the means of each row in a 2D dataset.

    Parameters:
    - `data` (np.ndarray): Input 2D dataset, potentially containing NaN values.
    - `alpha` (float, optional): Significance level (1 - confidence level). Defaults to 0.05 (95% confidence).
    - `return_means` (bool, optional): Whether to include the row means in the output. Defaults to True.

    Returns:
    - `tuple`: 
        - If `return_means` is True: `(row_means, lower_bounds, upper_bounds)`
        - If `return_means` is False: `(lower_bounds, upper_bounds)`
        
        Where:
        - `row_means` (np.ndarray): Means of each row, ignoring NaNs.
        - `lower_bounds` (np.ndarray): Lower bounds of the confidence intervals for each row.
        - `upper_bounds` (np.ndarray): Upper bounds of the confidence intervals for each row.
    """

    # Compute row-wise statistics, ignoring NaNs
    with np.errstate(invalid='ignore'):  # Suppress warnings for NaNs in mean/std
        row_means = np.nanmean(data, axis=1)
        row_stds = np.nanstd(data, ddof=1, axis=1)
    
    # Count non-NaN observations per row
    row_ns = np.sum(~np.isnan(data), axis=1)
    
    # Calculate standard error for each row
    row_std_errs = row_stds / np.sqrt(row_ns)
    
    # Determine whether to use the normal distribution (for large samples) or the t-distribution
    use_normal = row_ns >= 30

    # Initialize array to store Z-scores or T-scores
    scores = np.empty_like(row_ns, dtype=float)
    
    # Calculate scores based on the desired confidence level
    scores[use_normal] = stats.norm.ppf(1 - alpha / 2)  # Corrected to use 1 - alpha/2 for two-tailed test
    scores[~use_normal] = stats.t.ppf(1 - alpha / 2, row_ns[~use_normal] - 1)  # T-scores for small samples

    # Compute confidence interval margins for each row
    ci_margins = scores * row_std_errs
    
    # Compute lower and upper bounds of the confidence intervals for each row
    lower_bounds = row_means - ci_margins
    upper_bounds = row_means + ci_margins

    if return_means:
        return row_means, lower_bounds, upper_bounds
    else:
        return lower_bounds, upper_bounds

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

# def _convert_to_integer_season(s: float): # TODO: analyse brute force detector to mimic a cinversion method
#     """
#     Convert a decimal seasonal period to an integer by expressing it as a fraction,
#     reducing it to its simplest form, and returning the numerator.

#     Args:
#     - s (float): Decimal seasonal period.

#     Returns:
#     - int: Numerator of the reduced fraction representing the seasonal period.
#     """
#     # Convert decimal to fraction
#     frac = Fraction(s).limit_denominator()

#     # Return the numerator
#     return frac.numerator

# def _create_merged_dict(tuples_list): # NOT USED NOW
#     result = {}
    
#     for tup in tuples_list:
#         if len(tup) != 2:
#             raise ValueError(f"Invalid tuple format: {tup}")
        
#         key, value = tup
        
#         if isinstance(value, (int, float)):
#             if key in result:
#                 result[key].append(value)
#             else:
#                 result[key] = [value]
#         else:
#             raise TypeError(f"Value must be a number: {value}")
    
#     return result