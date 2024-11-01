import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks
from scipy.stats import chi2
from..utils import stationarize, compute_cartesian_products

def fft_seasonality(
    data: ArrayLike,
    alpha: float = 0.05,
    plot_results: bool = False,
    seasonality_type: str = 'auto',
    apply_cartesian: bool = False
) -> np.ndarray:
    """
    Detect seasonality using the Fast Fourier Transform (FFT) and peak detection.

    Args:
    - data (ArrayLike): Input time series data.
    - alpha (float, optional): Significance level for the chi-squared test. Defaults to 0.05.
    - plot_results (bool, optional): Whether to display the power spectrum plot. Defaults to False.

    Returns:
    - np.ndarray: Array of detected seasonal periods (1 / significant frequencies).
    """

    # Stationarize data
    ts, _, seasonality_type = stationarize(data=data, seasonality_type=seasonality_type)
    
    # Handle NA
    if seasonality_type == 'multiplicative':
        ts = np.nan_to_num(x=ts, nan=0)
        
    # Compute FFT
    fourier = np.fft.rfft(ts)
    psd = np.abs(fourier)**2 / len(ts)
    frequencies = np.fft.rfftfreq(len(ts))

    # Threshold for significant peaks (chi-squared test)
    threshold = chi2.ppf(1 - alpha, df=2) / 2 * np.mean(psd)

    # Detect peaks in the power spectrum
    peaks = find_peaks(psd, height=threshold)[0]

    # Plot results (if enabled)
    if plot_results:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(frequencies, psd, label="Power Spectrum")
        plt.fill_between(frequencies, threshold, alpha=0.3, color="grey")
        plt.scatter(frequencies[peaks], psd[peaks], color="red", label="Significant Frequencies")
        plt.legend()
        plt.show()

    # Return detected seasonal periods
    return (seasonality_type, compute_cartesian_products(1 / frequencies[peaks])) if apply_cartesian else (seasonality_type, 1 / frequencies[peaks])

