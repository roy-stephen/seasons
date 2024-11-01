import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import chi2
import matplotlib.pyplot as plt
from..utils import stationarize, compute_cartesian_products

def welch_seasonality(
    data: np.typing.ArrayLike,
    alpha: float = 0.05,
    plot_results: bool = True,
    seasonality_type: str = 'auto',
    apply_cartesian: bool = False
) -> np.ndarray:
    """
    Detect seasonality using Welch's periodogram, a modified Fourier analysis technique.

    **Welch's Periodogram:**
    Welch's method is a non-parametric spectral estimation technique that reduces the variance of the periodogram, a Fourier-based power spectral density (PSD) estimator. The periodogram is calculated by dividing the time series into overlapping segments, computing the Fourier transform for each segment, and then averaging the squared magnitudes of the transforms. This approach provides a more robust estimate of the PSD, especially for shorter time series.

    Args:
    - data (ArrayLike): Input time series data.
    - alpha (float, optional): Significance level for the chi-squared test. Defaults to 0.05.
    - plot_results (bool, optional): Whether to display the periodogram plot. Defaults to True.

    Returns:
    - np.ndarray: Array of detected seasonal periods (1 / significant frequencies).
    """

    # Stationarize data
    ts, _, seasonality_type = stationarize(data=data, seasonality_type=seasonality_type)
    
    # Handle NA
    if seasonality_type == 'multiplicative':
        ts = np.nan_to_num(x=ts, nan=0)
        
    # Compute Welch's periodogram
    N = len(data)
    nperseg = N // 3
    frequencies, ps = welch(
        x=ts,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )

    # Plot results (if enabled)
    if plot_results:
        plt.figure()
        plt.semilogy(frequencies, ps)
        # Compute significance
        dof = 2 * nperseg
        threshold = chi2.ppf(1 - alpha, dof) / dof * np.mean(ps)
        # Identify significant frequencies
        peaks = find_peaks(x=ps, height=threshold)[0]
        # Highlight significant frequencies on the plot
        plt.fill_between(x=frequencies, y1=threshold, color="grey", alpha=0.5, label=f"{1-alpha:.0%} Significance Level")
        plt.scatter(x=frequencies[peaks], y=ps[peaks], color="r")
        plt.legend()
        plt.show()

    # Return detected seasonal periods
    return (seasonality_type, compute_cartesian_products(1 / frequencies[peaks])) if apply_cartesian else (seasonality_type, 1 / frequencies[peaks])
