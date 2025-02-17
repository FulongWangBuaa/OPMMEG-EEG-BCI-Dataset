from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple
from scipy import signal
import numpy as np

def myfilterbank1(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 1) -> ndarray:
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

def myfilterbank2(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 2) -> ndarray:
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

def myfilterbank3(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 3) -> ndarray:
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

def myfilterbank4(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 4) -> ndarray:
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

def myfilterbank5(dataself,
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

def myfilterbank6(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 6) -> ndarray:
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


def myfilterbank7(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 7) -> ndarray:
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

def myfilterbank8(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 8) -> ndarray:
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


def myfilterbank9(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 9) -> ndarray:
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


def myfilterbank10(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 10) -> ndarray:
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