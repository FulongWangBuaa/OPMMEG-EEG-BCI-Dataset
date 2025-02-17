import numpy as np

def normalize_array(arr, axis=0):
    '''
    axis=0: 按列归一化
    axis=1: 按行归一化
    axis=None: 按整个数组归一化
    '''
    # 计算指定轴上的最小值和最大值  
    min_val = np.min(arr, axis=axis, keepdims=True)  
    max_val = np.max(arr, axis=axis, keepdims=True)  

    normalized_arr = (arr - min_val) / (max_val - min_val)

    return normalized_arr

def find_nearest_index(array, values):
    array = np.asarray(array)
    if not isinstance(values, list):
        index=(np.abs(array - values)).argmin()
        return index
    else:
        index = []
        for value in values:
            index.append((np.abs(array - value)).argmin())
        return index

def find_max_and_index(lst):  
    if len(lst) == 0:
        return None, None  
    max_value = lst[0]  
    max_index = 0  
    for index, value in enumerate(lst):  
        if value > max_value:  
            max_value = value  
            max_index = index  
    return max_value, max_index 


import numpy as np
from scipy import signal


def amplitude_spectrum(dataself,X):
    sfreq = dataself.srate
    if X.ndim == 1:
        n = len(X)
        f = np.fft.fftfreq(n, 1/sfreq)  # 频率向量  
        Y = np.fft.fft(X)  # 计算FFT  
        amplitude_spectrum = np.abs(Y)/n*2  # 幅度谱
        return f[:n//2], amplitude_spectrum[:n//2]
    else:
        n = X.shape[-1]
        f = np.fft.fftfreq(n, 1/sfreq)  # 频率向量
        Y = np.fft.fft(X,axis=-1)
        amplitude_spectrum = np.abs(Y)/n*2  # 幅度谱
        return f[:n//2], amplitude_spectrum[...,:n//2]

def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    """
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd)

    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise


def get_reference_signal(targets,sfreq,num_harmonics,num_samples):
    if type(targets) is int or float:
        targets = [targets]
    reference_signals = []
    t = np.arange(0, (num_samples / sfreq), step=1.0 / sfreq)
    for f in targets:
        reference_f = np.zeros((num_harmonics*2,num_samples))
        for h in range(1, num_harmonics + 1):
            reference_f[(h-1)*2,:]=np.sin(2 * np.pi * h * f * t)[0:num_samples]
            reference_f[(h-1)*2+1,:]=np.cos(2 * np.pi * h * f * t)[0:num_samples]
        reference_signals.append(reference_f)
    reference_signals = np.asarray(reference_signals)
    return reference_f

def power_ratio(X, f, target, num_harmonics):
    freqs = [target * i for i in range(1, num_harmonics + 1)]
    freqs_idx = [np.argmin(np.abs(f - i)) for i in freqs]
    powers = [X[i] for i in freqs_idx]
    ratios = [power/np.max(powers) for power in powers]
    return ratios
