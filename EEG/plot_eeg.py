import numpy as np
import matplotlib.pyplot as plt

from EEG.config import FS, CUTOFFS


def plot_frequency_band(data, channel_index, patient_index, band_number=2):
    """
    Plots the frequency band.

    Args:
    data (ndarray): EEG data. Can be one-dimensional or two-dimensional.
    channel_index (int): Channel number.
    patient_index (int): Patient number.
    band_number (int): Frequency band number.

    """
    signal = data[patient_index][channel_index]

    if band_number >= len(CUTOFFS):
        raise ValueError(f'Maksymalny numer pasma: {len(CUTOFFS) - 1}')

    if signal.ndim == 1:
        frequencies = np.fft.fftfreq(len(signal), d=1 / FS)
        fft_values = np.fft.fft(signal)
        magnitude_spectrum = np.abs(fft_values)

        plt.plot(frequencies, magnitude_spectrum, label=f'{CUTOFFS[band_number]} Hz')
        plt.title('Widmo częstotliwościowe')
        plt.xlabel('Częstotliwość (Hz)')
        plt.ylabel('Amplituda')
        plt.legend()
        plt.show()
    else:
        raise ValueError("Nieprawidłowe wymiary danych. Dane powinny być jednowymiarowe.")


def plot_eeg_signal(data, channel_index, patient_index=None):
    """
    Plots the EEG signal.

    Args:
    data (ndarray): EEG data.
    channel_index (int): Channel number.
    patient_index (int, optional): Patient number. Required if the data is three-dimensional.
    """

    if patient_index is not None:
        t = np.arange(0, data[patient_index][channel_index].shape[0]) / FS
        signal = data[patient_index][channel_index]

        print(f"Ilość próbek: {data[patient_index][channel_index].shape[0]}")
        print(f"Czas: {t[-1]:.3f} s")

        plt.plot(t, signal, label=f'Pacjent {patient_index}, Kanał {channel_index}')

    elif patient_index is None:
        t = np.arange(0, data[channel_index].shape[0]) / FS
        signal = data[channel_index]

        print(f"Ilość próbek: {data[channel_index].shape[0]}")
        print(f"Czas: {t[-1]:.3f} s")

        plt.plot(t, signal, label=f'Kanał {channel_index}')

    else:
        raise ValueError("Nieprawidłowy wymiar danych. Dane powinny być dwuwymiarowe lub trójwymiarowe.")

    plt.xlabel('Czas (s)')
    plt.ylabel('Wartości próbek')
    plt.title('Wykres sygnału EEG')
    plt.legend()
    plt.show()
