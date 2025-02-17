# -*- coding: utf-8 -*-

from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple
from scipy import signal
import numpy as np

ch_names_meg = ['FP1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1',
           'C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7',
           'P9','PO7','PO3','O1','IZ','OZ','POZ','PZ','CPZ','FPZ','FP2',
           'AF8','AF4','AFZ','FZ','F2','F4','F6','F8','FT8','FC6','FC4',
           'FC2','FCZ','CZ','C2','C4','C6','T8','TP8','CP6','CP4','CP2',
           'P2','P4','P6','P8','P10','PO8','PO4','O2']

def meg_occipital_chs(chs):
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

    pick_ch_meg_idx = [ch_names_meg.index(pick_ch) for pick_ch in ch_used]
    return pick_ch_meg_idx

def meg_occipital_chs_snr(chs):
    if chs == 1:
        ch_used = ['O2']
    elif chs == 3:
        ch_used = ['O2','PO3','O1']
    elif chs == 5:
        ch_used = ['O2','PO3','O1','OZ','POZ']
    elif chs == 7:
        ch_used = ['O2','PO3','O1','OZ','POZ','PO7','PO4']
    elif chs == 9:
        ch_used = ['O2','PO3','O1','OZ','POZ','PO7','PO4','PO8','IZ']
    elif chs == 11:
        ch_used = ['O2','PO3','O1','OZ','POZ','PO7','PO4','PO8','IZ','P9','P10']
    elif chs == 13:
        ch_used = ['O2','PO3','O1','OZ','POZ','PO7','PO4','PO8','IZ','P9','P10',
                   'CP1','P1']
    elif chs == 15:
        ch_used = ['O2','PO3','O1','OZ','POZ','PO7','PO4','PO8','IZ','P9','P10',
                   'CP1','P1','P4','CPZ']
    elif chs == 17:
        ch_used = ['O2','PO3','O1','OZ','POZ','PO7','PO4','PO8','IZ','P9','P10',
                   'CP1','P1','P4','CPZ','CP2','P7']

    pick_ch_meg_idx = [ch_names_meg.index(pick_ch) for pick_ch in ch_used]
    return pick_ch_meg_idx

def suggested_weights_filterbank(num_subbands: Optional[int] = 5) -> List[float]:
    """
    Provide suggested weights of filterbank for benchmark dataset

    Returns
    -------
    weights_filterbank : List[float]
        Suggested weights of filterbank
    """
    return [i**(-1.25)+0.25 for i in range(1,num_subbands+1,1)]

def meg_suggested_ch() -> List[int]:
    """
    Provide suggested channels for benchmark dataset

    Returns
    -------
    ch_used: List
        Suggested channels (PZ, PO3, POz, PO4, O1, Oz, O2, Iz)
    """
    return [30, 25, 29, 62, 26, 28, 63, 27]

def meg_occipital_17_ch() -> List[int]:
    """
    Provide 19 channels around occipital region for benchmark dataset

    Returns
    -------
    ch_used: List
        17 channels in occipital regions (P7, P5, P3, P1, PZ, P2, P4, P6,
        P8, PO7, PO3, POZ, PO4, PO8, O1, OZ, O2)
    """
    return [22,21,20,19,30,56,57,58,59,24,25,29,62,61,26,28,63]

def meg_occipital_19_ch() -> List[int]:
    """
    Provide 19 channels around occipital region for benchmark dataset

    Returns
    -------
    ch_used: List
        19 channels in occipital regions (P7, P5, P3, P1, PZ, P2, P4, P6,
        P8, PO7, PO5, PO3, POz, PO4, PO6, PO8, O1, Oz, O2)
    """
    return [19,20,21,22,23,24,25,26,27,28,29,30,56,57,58,59,60,61,62,63]

def meg_center_occipital_26_ch() -> List[int]:
    """
    Provide 26 channels from center region to occipital region for benchmark dataset

    Returns
    -------
    ch_used: List
        26 channels from center region to occipital region 
        (CP5, CP3, CP1, CPZ, CP2, CP4, CP6,
         P7, P5, P3, P1, PZ, P2, P4, P6, P8,
         PO7, PO5, PO3, POz, PO4, PO6, PO8
         O1, Oz, O2)
    """
    return [33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59]

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