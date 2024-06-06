import numpy as np
import matplotlib.pyplot as plt


def plot_frequency_band(data, channel_index, patient_index, band_number=2):
    """
    Plots the frequency band.

    Args:
    data (ndarray): EEG data. Can be one-dimensional or two-dimensional.
    channel_index (int): Channel number.
    patient_index (int): Patient number.
    band_number (int): Frequency band number.

    """
    from EEG.config import FS, CUTOFFS
    signal = data[patient_index][channel_index]

    if band_number >= len(CUTOFFS):
        raise ValueError(f'Maximum band number: {len(CUTOFFS) - 1}')

    if signal.ndim == 1:
        frequencies = np.fft.fftfreq(len(signal), d=1 / FS)
        fft_values = np.fft.fft(signal)
        magnitude_spectrum = np.abs(fft_values)

        plt.plot(frequencies, magnitude_spectrum, label=f'{CUTOFFS[band_number]} Hz')
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
    else:
        raise ValueError("Invalid data dimensions. Data should be one-dimensional.")


def plot_eeg_signal(data, channel_index, patient_index=None):
    """
    Plots the EEG signal.

    Args:
    data (ndarray): EEG data.
    channel_index (int): Channel number.
    patient_index (int, optional): Patient number. Required if the data is three-dimensional.
    """
    from EEG.config import FS
    if patient_index is not None:
        t = np.arange(0, data[patient_index][channel_index].shape[0]) / FS
        signal = data[patient_index][channel_index]

        print(f"Number of samples: {data[patient_index][channel_index].shape[0]}")
        print(f"Time: {t[-1]:.3f} s")

        plt.plot(t, signal, label=f'Patient {patient_index}, Channel {channel_index}')

    elif patient_index is None:
        t = np.arange(0, data[channel_index].shape[0]) / FS
        signal = data[channel_index]

        print(f"Number of samples: {data[channel_index].shape[0]}")
        print(f"Time: {t[-1]:.3f} s")

        plt.plot(t, signal, label=f'Channel {channel_index}')

    else:
        raise ValueError("Invalid data dimension. Data should be two-dimensional or three-dimensional.")

    plt.xlabel('Time (s)')
    plt.ylabel('Sample values')
    plt.title('EEG Signal Plot')
    plt.legend()
    plt.show()

