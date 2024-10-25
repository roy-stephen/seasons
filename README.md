**Seasons: Time Series Seasonality Detection**
=============================================

**Overview**
------------

`seasons` is a Python package designed to detect seasonality in time series data. It provides a suite of algorithms to identify periodic patterns, enabling you to better understand and forecast your data.

**Key Features**
----------------

*   **Multiple Detection Algorithms**:
    *   **Brute Force Seasonality Detection**: Exhaustive search for seasonal periods.
    *   **FFT-Based Seasonality Detection**: Fast Fourier Transform (FFT) for efficient frequency analysis.
    *   **Welch's Periodogram Seasonality Detection**: Modified periodogram for robust power spectral density estimation.
*   **Stationarization and Preprocessing**:
    *   **Automatic Stationarization**: Ensure your time series data is stationary before analysis.
    *   **Flexible Data Import**: Work with various time series data formats.
*   **Visualization and Interpretation**:
    *   **Interactive Plots**: Visualize detected seasonalities and their corresponding frequencies. (TODO)
    *   **Clear Documentation**: Understand the underlying algorithms and their applications.

**Getting Started**
-------------------

### Installation

Install `seasons` using pip: (Not working yet)
```bash
pip install seasons
```
For the latest development version, install from TestPyPI:
```bash
pip install -i https://test.pypi.org/simple/ seasons
```
### Example Usage

```python
import pandas as pd
from seasons import detection, utils

# Load sample time series data
df = pd.read_csv('sample_data.csv', index_col='date', parse_dates=['date'])

# Detect seasonality using FFT
seasonal_periods = detection.fft_seasonality(df['value'])

# Visualize the detected seasonality (TODO)
detection.plot_seasonality(stationary_data, seasonal_periods)
```
**Algorithms and Techniques**
-----------------------------

### Brute Force Seasonality Detection

*   **Description**: Exhaustive search for seasonal periods by iterating over possible frequencies.
*   **Use Case**: Small to medium-sized datasets where computational efficiency is not a primary concern.

### FFT-Based Seasonality Detection

*   **Description**: Utilizes the Fast Fourier Transform (FFT) to efficiently analyze frequencies in the time series data.
*   **Use Case**: Large datasets where computational efficiency is crucial.

### Welch's Periodogram Seasonality Detection

*   **Description**: Modified periodogram for robust power spectral density estimation, reducing noise and providing more accurate results.
*   **Use Case**: Datasets with significant noise or non-stationarities.

**API Documentation**
----------------------

### `seasons.detection`

*   `brute_force_seasonality(data, alpha=0.05)`: Brute force seasonality detection.
*   `fft_seasonality(data, alpha=0.05)`: FFT-based seasonality detection.
*   `welch_seasonality(data, alpha=0.05)`: Welch's periodogram seasonality detection.

### `seasons.utils`

*   `stationarize(data)`: Automatic stationarization of time series data.
*   `series_generator(length, seasonality_periods, noise_level)`: Generate synthetic time series data with specified seasonality and noise.

**Contributing**
------------

*   Fork the repository and submit a pull request with your changes.
*   Ensure all changes are accompanied by relevant tests and documentation updates.

**License**
-------

`seasons` is released under the [MIT License](LICENSE).

**Acknowledgments**
----------------

*   Inspired by various time series analysis libraries and research papers.
*   Special thanks to the open-source community for their contributions and support.