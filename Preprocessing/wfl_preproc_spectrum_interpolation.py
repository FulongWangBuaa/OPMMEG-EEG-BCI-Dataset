import numpy as np
import mne
from mne.io.pick import _picks_to_idx
import copy

def spectrum_interpolation(raw,Fl,bandwidth,neighbourwidth,exclude=None):
    # defaults
    # 获取函数参数中的可选参数
    Fs = raw.info['sfreq']
    picks = None
    if exclude is not None:
        picks = _picks_to_idx(raw.info, picks, exclude=exclude)
    else:
        picks = _picks_to_idx(raw.info, picks, exclude=())
    picks_good, picks_bad = list(), list()  # these are indices into picks
    for ii, pi in enumerate(picks):
        if raw.ch_names[pi] in raw.info["bads"]:
            picks_bad.append(ii)
        else:
            picks_good.append(ii)
    picks_good = np.array(picks_good, int)
    picks_bad = np.array(picks_bad, int)

    dat = raw.get_data()[picks_good,:]

    nchans, nsamples = dat.shape
    # 检查输入数据是否包含 NaN 值，如果存在则输出警告信息
    if np.any(np.isnan(dat)):
        print('data contains NaN values')
    # 将输入的频率Fl转换为一维数组
    Fl = np.atleast_1d(Fl)
    # 计算在给定采样率Fs下，能够容纳完整线噪声周期的整数个周期数
    n = np.round(np.floor(nsamples * (Fl / Fs + 100 * np.finfo(float).eps)) * Fs / Fl)
    # 如果数据不能在单个步骤中滤波，则递归地为每个频率执行滤波操作，
    # 并将结果级联处理
    if np.size(n) > 1 or not np.all(n == n[0]):
        filt = raw.copy()
        for i in range(np.size(Fl)):
            filt = spectrum_interpolation(raw=filt, Fl=Fl[i],bandwidth=bandwidth[i], neighbourwidth=neighbourwidth[i])
        return filt

    # 检查数据长度是否与线噪声频率的完整周期匹配
    if np.size(Fl) < np.size(bandwidth):
        bandwidth = bandwidth[:np.size(Fl)]
    if np.size(Fl) < np.size(neighbourwidth):
        neighbourwidth = neighbourwidth[:np.size(Fl)]
    if n != nsamples:
        raise ValueError('Spectrum interpolation requires that the data length fits complete cycles of the powerline frequency.')

    nfft = nsamples
    if np.size(Fl) != np.size(bandwidth) or np.size(Fl) != np.size(neighbourwidth):
        raise ValueError('The number of frequencies to interpolate should be the same as the number of bandwidths and neighbourwidths')
    # frequencies to interpolate
    f2int = np.array([Fl - bandwidth, Fl + bandwidth]).T

    # frequencies used for interpolation
    f4int = np.array([f2int[:, 0] - neighbourwidth, f2int[:, 0], f2int[:, 1], f2int[:, 1] + neighbourwidth]).T

    data_fft = np.fft.fft(dat, n=nfft, axis=1)

    frq = np.linspace(start = 0, stop = 1, num = nfft+1) * Fs

    # interpolate 50Hz (and harmonics) amplitude in spectrum
    for i in range(np.size(Fl)):
        # samples of frequencies that will be interpolated
        smpl2int = np.arange(np.argmin(np.abs(frq - f2int[i, 0])),
                                np.argmin(np.abs(frq - f2int[i, 1])) + 1)

        # samples of neighbouring frequencies used to calculate the mean
        low_neighbouring = np.arange(np.argmin(np.abs(frq - f4int[i, 0])),np.argmin(np.abs(frq - f4int[i, 1])))
        high_neighbouring = np.arange(np.argmin(np.abs(frq - f4int[i, 2])) + 1, np.argmin(np.abs(frq - f4int[i, 3])) + 1)

        smpl4int = np.concatenate((low_neighbouring,high_neighbouring))

        # new amplitude is calculated as the mean of the neighbouring frequencies
        mns4int = np.ones(data_fft[:,smpl2int].shape) * np.mean(np.abs(data_fft[:, smpl4int]), axis=1,keepdims=True)

        # Eulers formula: replace noise components with new mean amplitude combined with phase, that is retained from the original data
        data_fft[:, smpl2int] = np.exp(1j * np.angle(data_fft[:, smpl2int])) * mns4int
    # complex fourier coefficients are transformed back into time domin, fourier coefficients are treated as conjugate 'symmetric'
    # to ensure a real valued signal after iFFT
    half_len = data_fft.shape[1] // 2
    data_fft = np.column_stack((data_fft[:,0],data_fft[:,1:half_len],data_fft[:,half_len],np.conj(np.flip(data_fft[:,1:half_len],axis=1))))
    filt = np.fft.ifft(data_fft, axis=1)
    filt = filt.real

    raw_interpolation = raw.copy()
    raw_interpolation._data[picks_good,:] = filt

    return raw_interpolation
