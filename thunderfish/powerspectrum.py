import numpy as np
import matplotlib.mlab as ml

def calc_psd(data, samplerate, fresolution):
    """
    Calculates a Powerspecturm.

    This function takes a data array, its samplerate and a frequencyresolution for the powerspectrum.
    With this input it first calculates a nfft value and later a powerspectrum.

    :param data: (1-D array) data array you want to calculate a psd of.
    :param samplerate: (float) sampling rate of the data that you want to calculate a psd of.
    :param fresolution: (float) frequency resolution of the psd.
    :return power:(1-D array) power array of the psd.
    :return freqs: (1-D array) psd array of the psd.
    """

    nfft = int(np.round(2 ** (np.floor(np.log(samplerate / fresolution) / np.log(2.0)) + 1.0)))
    if nfft < 16:
        nfft = 16
    power, freqs = ml.psd(data, NFFT=nfft, noverlap=nfft / 2, Fs=samplerate, detrend=ml.detrend_mean)
    return power, freqs

def powerspectrumplot(power, freqs, ax):
    """
    Plots a powerspectum.

    :param power:               (1-D array) power array of a psd.
    :param freqs:               (1-D array) frequency array of a psd.
    :param ax:                  (axis for plot) empty axis that is filled with content in the function.
    :return ax:                 (axis for plot) axis that is ready for plotting containing the powerspectrum.
    """
    ax.plot(freqs, 10.0 * np.log10(power))
    ax.set_ylabel('power [dB]')
    ax.set_xlabel('frequency [Hz]')
    ax.set_xlim([0, 3000])
    return ax

def powerspectrum(data, samplingrate, fresolution=0.5, plot_data_func=None, **kwargs):
    """
    This function is performing the steps to calculate a powerspectrum on the basis of a given dataset, a given
    samplingrate and a given frequencyresolution for the psd. Therefore two other functions are called to first
    calculate the nfft value and second calculate the powerspectrum.

    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplingrate:        (float) sampling rate of the data that you want to calculate a psd of.
    :param fresolution:         (float) frequency resolution of the psd
    :param plot_data_func:      (function) function (powerspectrumplot()) that is used to create a axis for later plotting containing the calculated powerspectrum.
    :param **kwargs:            additional arguments that are passed to the plot_data_func().
    :return power:              (1-D array) power array of the psd.
    :return freqs:              (1-D array) psd array of the psd.
    :return ax:                 (axis for plot) axis that is ready for plotting containing a figure that shows what the modul did.
    """

    power, freqs = calc_psd(data, samplingrate, fresolution)

    if plot_data_func:
        ax = plot_data_func(power, freqs, **kwargs)
        return power, freqs, ax
    else:
        return power, freqs

if __name__ == '__main__':

    print('Computes powerspectrum of a created signal of two wavefish (300 and 450 Hz)')
    print('')
    print('Usage:')
    print('  python powerspectrum.py')
    print('')

    fundamental = [300, 450] # Hz
    samplingrate = 100000
    time = np.linspace(0, 8-1/samplingrate, 8*samplingrate)
    data = np.sin(time * 2 * np.pi* fundamental[0]) + np.sin(time * 2 * np.pi* fundamental[1])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    power, freqs, ax = powerspectrum(data, samplingrate, plot_data_func=powerspectrumplot, ax=ax)
    plt.show()
