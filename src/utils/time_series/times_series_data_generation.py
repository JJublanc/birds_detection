import numpy as np


def trend(time: np.array, slope: float = 0):
    """Add a trend to a series of values.
    :param time: series of evenly spaced values
    :param slope: amplitude of the trend
    :return: series of values with a trend
    """
    return slope * time


def seasonal_pattern_1(season_time: float):
    """Arbitrary pattern to be seasonally repeated.
    :param season_time: series of values seasonal values between 0 and 1
    :return: pattern applied to the series
    """
    return np.where(
        season_time < 0.4,
        np.cos(season_time * 2 * np.pi),
        1 / np.exp(3 * season_time),
    )


def seasonal_pattern_2(season_time):
    """Arbitrary pattern to be seasonally repeated.
    :param season_time: series of values seasonal values between 0 and 1
    :return: pattern applied to the series
    """
    return np.sin(season_time * 20 * np.pi)


def seasonality(seasonal_pattern, time, period, amplitude=1, phase=0):
    """
    Get a seasonal values from a series of consecutive integers
    :param seasonal_pattern: pattern to be applied seasonally
    :param time: series of consecutive integers
    :param period: period in which the pattern is repeated
    :param amplitude: scale to which values are amplified
    :param phase: allow an offset of the values
    :return: seasonal values
    """
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    """
    generate a white noise series of successive integers
    :param time: series of consecutive integers
    :param noise_level: amplitude of the noise
    :param seed: set a seed to make results reproducible
    :return: white noise series
    """
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def autocorrelation(
    time,
    amplitude,
    phi,
    lags,
    seed=None,
):
    """
    generate a auto-correlated series from two series consecutive integers
    :param time: series of consecutive integers
    :param amplitude: scale to which values are amplified
    :param phi: dictionary of the 4 auto-correlation coefficients
    :param lags: dictionary of the 4 auto-correlation lags
    :param seed: set a seed to make results reproducible
    :return: two auto-correlated series
    """
    rnd_1 = np.random.RandomState(seed[0])
    rnd_2 = np.random.RandomState(seed[1])

    max_lag = np.max(list(lags.values()))
    ar_1 = rnd_1.randn(len(time) + max_lag)
    ar_2 = rnd_2.randn(len(time) + max_lag)

    ar_1[:max_lag] = 1
    for step in range(max_lag, len(time) + max_lag):
        ar_1[step] += phi["phi1_1"] * ar_1[step - lags["lag1_1"]]
        ar_2[step] += phi["phi2_1"] * ar_1[step - lags["lag2_1"]]
        ar_1[step] += phi["phi1_2"] * ar_2[step - lags["lag1_2"]]
        ar_2[step] += phi["phi2_2"] * ar_2[step - lags["lag2_2"]]
    return ar_1[max_lag:] * amplitude, ar_2[max_lag:] * amplitude * 0.2
