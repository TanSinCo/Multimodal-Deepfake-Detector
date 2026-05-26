
import numpy as np

from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=4):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype='band'
    )

    return b, a


def apply_bandpass_filter(signal, fs=30):

    if len(signal) <= 27:
        return signal

    b, a = butter_bandpass(

        lowcut=0.7,
        highcut=4.0,

        fs=fs,
        order=4
    )

    filtered = filtfilt(b, a, signal)

    return filtered


def zscore_normalize(signal):

    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-6:
        return signal

    normalized = (signal - mean) / std

    return normalized


def filter_rppg(rppg_signal):

    filtered_regions = []

    num_regions = rppg_signal.shape[1]

    for i in range(num_regions):

        signal = rppg_signal[:, i]

        filtered = apply_bandpass_filter(
            signal,
            fs=30
        )

        normalized = zscore_normalize(filtered)

        filtered_regions.append(normalized)

    filtered_regions = np.stack(
        filtered_regions,
        axis=1
    )

    return filtered_regions
