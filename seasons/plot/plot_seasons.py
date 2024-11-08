import numpy as np
import matplotlib.pyplot as plt
# from fractions import Fraction
from scipy import stats
from..detect import brute_force_seasonality
from..utils import remove_trend
from..utils.estimate_effect import _confidence_interval, _plot_seasonal_components, _repeat_array_until_length


def plot_seasonal_components(
    ts: np.typing.ArrayLike,
    seasons: np.typing.ArrayLike = None,
    seasonality_type: str = 'auto',
    normalize: bool = True,
    alpha: float = 0.05
):
    
    # Check if all items of seasons are integer # TODO: allow for decimal seasons next
    if seasons: # if the user specify some seasonal periods
        if not all(isinstance(s, int) for s in seasons):
            print("Decimal seasonality periods are not currenytly supported. Will round to integer.")
            seasons = [int(s) for s in seasons]

            # Detrend
            trend, detrended, seasonality_type = remove_trend(data=ts, seasonality_type=seasonality_type)
            N = len(detrended)
            
            # estimating effects
            c, seasons_effects = _confidence_interval(data=detrended, seasonal_periods=seasons, normalize=normalize, seasonality_type=seasonality_type, alpha=alpha)
            if normalize:
                if seasonality_type == 'additive':
                    trend -= abs(c)
                    detrended = ts - trend
                else:
                    trend *= c
                    detrended = ts / trend

            _plot_seasonal_components(
                seasonal_components=seasons_effects,
                data=ts,
                trend=trend,
                detrended=detrended,
                seasonality_type=seasonality_type
            )

    else: # if the user don't specify any seasons
        print("No seasons were specified.\nUsing bruteforce to estimate seasonal periods...")
        seasonality_type, seasons = brute_force_seasonality(
            data=ts, 
            alpha=alpha, 
            seasonality_type=seasonality_type,
            display_plot=True
        )
        print(f"Detected seasons: {seasons}.")
    
