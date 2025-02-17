# -*- coding: utf-8 -*-

from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple
from scipy import signal
import numpy as np

ch_names_eeg = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3',
           'F1','FZ','F2','F4','F6','F8','FC5', 'FC3',
           'FC1','FCZ','FC2','FC4','FC6','T7','C5',
           'C3','C1','CZ','C2','C4','C6','T8','TP7','CP5',
           'CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7',
           'P5','P3','P1','PZ','P2','P4','P6','P8','PO7',
           'PO3','POZ','PO4','PO8','O1','OZ','O2']

def eeg_occipital_chs(chs):
    if chs == 1:
        ch_used = ['OZ']
    elif chs == 3:
        ch_used = ['OZ','POZ','PZ']
    elif chs == 5:
        ch_used = ['OZ','POZ','PZ','O1','O2']
    elif chs == 7:
        ch_used = ['OZ','POZ','PZ','O1','O2','PO3','PO4']
    elif chs == 9:
        ch_used = ['OZ','POZ','PZ','O1','O2','PO3','PO4','PO7','PO8']
    elif chs == 11:
        ch_used = ['OZ','POZ','PZ','O1','O2','PO3','PO4','PO7','PO8','P1','P2']
    elif chs == 13:
        ch_used = ['OZ','POZ','PZ','O1','O2','PO3','PO4','PO7','PO8','P1','P2',
                   'P3','P4']
    elif chs == 15:
        ch_used = ['OZ','POZ','PZ','O1','O2','PO3','PO4','PO7','PO8','P1','P2',
                   'P3','P4','P5','P6']
    elif chs == 17:
        ch_used = ['OZ','POZ','PZ','O1','O2','PO3','PO4','PO7','PO8','P1','P2',
                   'P3','P4','P5','P6','P7','P8']
    pick_ch_eeg_idx = [ch_names_eeg.index(pick_ch) for pick_ch in ch_used]
    return pick_ch_eeg_idx

def eeg_occipital_chs_snr(chs):
    if chs == 1:
        ch_used = ['POZ']
    elif chs == 3:
        ch_used = ['POZ','OZ','PO4']
    elif chs == 5:
        ch_used = ['POZ','OZ','PO4','PO3','O2']
    elif chs == 7:
        ch_used = ['POZ','OZ','PO4','PO3','O2','O1','PO7']
    elif chs == 9:
        ch_used = ['POZ','OZ','PO4','PO3','O2','O1','PO7','P2','P4']
    elif chs == 11:
        ch_used = ['POZ','OZ','PO4','PO3','O2','O1','PO7','P2','P4','PO8','P1']
    elif chs == 13:
        ch_used = ['POZ','OZ','PO4','PO3','O2','O1','PO7','P2','P4','PO8','P1',
                   'PZ','P3']
    elif chs == 15:
        ch_used = ['POZ','OZ','PO4','PO3','O2','O1','PO7','P2','P4','PO8','P1',
                   'PZ','P3','P5','P6']
    elif chs == 17:
        ch_used = ['POZ','OZ','PO4','PO3','O2','O1','PO7','P2','P4','PO8','P1',
                   'PZ','P3','P5','P6','P7','CP2']
    pick_ch_eeg_idx = [ch_names_eeg.index(pick_ch) for pick_ch in ch_used]
    return pick_ch_eeg_idx

def suggested_weights_filterbank(num_subbands: Optional[int] = 5) -> List[float]:
    """
    Provide suggested weights of filterbank for benchmark dataset

    Returns
    -------
    weights_filterbank : List[float]
        Suggested weights of filterbank
    """
    return [i**(-1.25)+0.25 for i in range(1,num_subbands+1,1)]

def eeg_suggested_ch() -> List[int]:
    """
    Provide suggested channels for benchmark dataset

    Returns
    -------
    ch_used: List
        Suggested channels (PZ, PO7, PO3, POz, PO4, PO8, O1, Oz, O2)
    """
    return [43, 48, 49, 50, 51, 52, 53, 54, 55]

def eeg_occipital_17_ch() -> List[int]:
    """
    Provide 19 channels around occipital region for benchmark dataset

    Returns
    -------
    ch_used: List
        19 channels in occipital regions (P7, P5, P3, P1, PZ, P2, P4, P6,
        P8, PO7, PO3, POz, PO4, PO8, O1, Oz, O2)
    """
    return [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

def eeg_center_occipital_26_ch() -> List[int]:
    """
    Provide 26 channels from center region to occipital region for benchmark dataset

    Returns
    -------
    ch_used: List
        26 channels from center region to occipital region 
        (TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, TP8,
         P7, P5, P3, P1, PZ, P2, P4, P6, P8,
         PO7, PO3, POz, PO4, PO8
         O1, Oz, O2)
    """
    return [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]

def preprocess(dataself,
               X: ndarray) -> ndarray:
    """
    Suggested preprocessing function for benchmark dataset
    
    notch filter at 50 Hz
    """
    srate = dataself.srate

    # notch filter at 50 Hz
    f0 = 50
    Q = 35
    notchB, notchA = signal.iircomb(f0, Q, ftype='notch', fs=srate)
    preprocess_X = signal.filtfilt(notchB, notchA, X, axis = X.ndim-1, padtype='odd', padlen=3*(max(len(notchB),len(notchA))-1))

    return preprocess_X


def filterbank(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    nyq = dataself.srate / 2

    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
    passband = [8, 18, 28, 38, 48, 58, 68, 78]
    stopband = [6, 16, 26, 36, 46, 56, 66, 76]
    highcut_pass, highcut_stop = 78, 88
    for k in range(0, num_subbands, 1):
        Wp = [passband[k] / nyq, highcut_pass / nyq]
        Ws = [stopband[k] / nyq, highcut_stop / nyq]

        gstop = 20
        N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
        bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
        filterbank_X[k,:,:] = signal.filtfilt(bpB, bpA, X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                
    return filterbank_X

def myfilterbank(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))

    for k in range(1, num_subbands+1, 1):
        Wp = [(8*k)/(srate/2), 90/(srate/2)]
        Ws = [(8*k-2)/(srate/2), 100/(srate/2)]

        gstop = 20
        N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
        bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
        filterbank_X[k-1,:,:] = signal.filtfilt(bpB,bpA,X,axis=1,padtype='odd',
                                                  padlen=3*(max(len(bpB),len(bpA))-1))
    return filterbank_X


def bandpass_filter(dataself, X: ndarray, lowcut, highcut, order=5):
    nyq = dataself.srate / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    if X.ndim == 1:
        Y = signal.filtfilt(b, a, X)
    else:
        Y = signal.filtfilt(b, a, X,axis=X.ndim-1)
    return Y