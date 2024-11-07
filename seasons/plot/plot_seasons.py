import numpy as np
import matplotlib.pyplot as plt
# from fractions import Fraction
from scipy import stats
from..detect import brute_force_seasonality
from..utils import remove_trend
from..utils.estimate_effect import _confidence_interval, _reshape_to_2d, _repeat_array_until_length


def plot_seasonal_components(
    series: np.typing.ArrayLike,
    seasons: np.typing.ArrayLike = None,
    seasonality_type: str = 'auto',
    repeat_seasons: bool = False,
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

    # Sort seasons
    seasons = np.sort(seasons)
    
    # Detrend
    detrended, seasonality_type = remove_trend(data=series, seasonality_type=seasonality_type)
    N = len(detrended)
    
    # Initialize plot
    N_FIG = len(seasons) + 4 # 4 spaces for the orignal series, the stationary series, the sum of the seasonal effects and the residuals
    fig, ax = plt.subplots(nrows=N_FIG, ncols=1, figsize=(16, 6*N_FIG))
    # Plot series
    ax[0].plot(series, label="Original series")
    ax[0].legend()
    ax[1].plot(detrended, label=f"Detrended series, Seas. type={seasonality_type}")
    ax[1].legend()

    # initialize total seasonal effect
    if seasonality_type == 'additive':
        total_seasonal_effect = 0
    else:
        total_seasonal_effect = 1

    # Estimate seasonal average for each seasons
    for i, s in enumerate(seasons):
        if seasonality_type == 'additive':
            partial_deseasonalized = detrended - total_seasonal_effect
        else:
            partial_deseasonalized = detrended / total_seasonal_effect
        # Reshape
        reshaped = _reshape_to_2d(partial_deseasonalized, s)
        # Compute means for seasonal effect estimation and stats for confidence interval
        avg, lower_bound, upper_bound = _confidence_interval(data=reshaped)
        # Harmonize size
        if repeat_seasons:
            avg = _repeat_array_until_length(avg, N)
            lower_bound = _repeat_array_until_length(lower_bound, N)
            upper_bound = _repeat_array_until_length(upper_bound, N)
        # Update total seasonal effect
        if seasonality_type == 'additive':
            total_seasonal_effect = total_seasonal_effect + avg
        else:
            partial_deseasonalized = total_seasonal_effect * avg
        # Plot component
        ax[i+4].plot(avg, marker='o', label=f"Seasonal Component s={s}")
        ax[i+4].fill_between(
            range(len(upper_bound)),
            upper_bound,
            lower_bound,
            color="grey",
            alpha=0.3,
            label="Confidence Bound"
        )
        ax[i+4].legend()

    # Plot total seasonal effect
    ax[2].plot(total_seasonal_effect, label=f"Total Seasonal Effect")
    ax[2].legend()
    # Plot residuals
    if seasonality_type == 'additive':
        residuals = detrended - total_seasonal_effect
    else:
        residuals = detrended / total_seasonal_effect
    ax[3].plot(residuals, label=f"Residuals")
    ax[3].legend()

    # Render plot
    plt.tight_layout()
    plt.suptitle("Seasonal Components")
    plt.show()

    return None