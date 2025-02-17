import numpy as np
import warnings
from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple, cast
from scipy.sparse import block_diag, identity, vstack, spmatrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eig, pinv, qr
from scipy import signal
from functools import partial
from joblib import Parallel, delayed
from copy import deepcopy
import scipy

from SSVEPAnalysisToolbox.algorithms.basemodel import BaseModel
from SSVEPAnalysisToolbox.algorithms.utils import (
    gen_template, sort, canoncorr, separate_trainSig, qr_list, 
    blkrep, eigvec, cholesky,inv, repmat
)


def filterFGx(data,srate,f,fwhm):
    '''
    :: filterFGx   Narrow-band filter via frequency-domain Gaussian
    filtdat,empVals]= filterFGx(data,srate,f,fwhm,showplot=0)


      INPUTS
         data : 1 X time or chans X time
        srate : sampling rate in Hz
            f : peak frequency of filter
         fhwm : standard deviation of filter, 
                defined as full-width at half-maximum in Hz
     showplot : set to true to show the frequency-domain filter shape (default=false)

      OUTPUTS
      filtdat : filtered data
      empVals : the empirical frequency and FWHM (in Hz and in ms)

    Empirical frequency and FWHM depend on the sampling rate and the
     number of time points, and may thus be slightly different from
     the requested values.

     mikexcohen@gmail.com
    '''

    ## compute filter

    n_channels,n_samples = data.shape

    # frequencies
    hz = np.linspace(0,srate,20*srate)

    # create Gaussian
    s  = fwhm*(2*np.pi-1)/(4*np.pi) # normalized width
    x  = hz-f                       # shifted frequencies
    fx = np.exp(-.5*(x/s)**2)       # gaussian
    fx = fx/np.max(fx)              # gain-normalized

    # apply the filter
    filtdat = np.zeros( np.shape(data) )
    for ci in range(filtdat.shape[0]):
        filtdat[ci,:] = 2*np.real( np.fft.ifft( np.fft.fft(data[ci,:],n=20*srate)*fx)[:n_samples] )

    return filtdat


def _ress_U(X: ndarray,
            stim_freq: float,
            filterbank_idx: int,
            srate: float,
            peakwidt: float,neighfreq: float,neighwidt: float) -> ndarray:
    """
    Calculate spatial filters of ress

    Parameters
    ------------
    X : list
        List of EEG data
        Length: (trial_num,)
        Shape of EEG: (channel_num, signal_len)

    Returns
    -----------
    U : ndarray
        Spatial filter
        shape: (channel_num * n_component)
    """
    n_trials = len(X)
    n_channels, n_samples = X[0].shape

    peakfreq = stim_freq * (filterbank_idx + 1)

    # compute covariance matrix at peak frequency
    fdatAt = np.zeros((n_channels, n_samples, n_trials))
    for ti in range(n_trials):
        tmdat = X[ti]
        fdatAt[:,:,ti] = filterFGx(tmdat,srate,peakfreq,peakwidt)
    fdatAt = fdatAt.reshape(n_channels, -1)
    fdatAt -= np.mean(fdatAt, axis=1, keepdims=True)
    covAt = np.dot(fdatAt, fdatAt.T) / n_samples

    # compute covariance matrix for lower neighbor
    fdatLo = np.zeros((n_channels, n_samples, n_trials))
    for ti in range(n_trials):
        tmdat = X[ti]
        fdatLo[:,:,ti] = filterFGx(tmdat,srate,peakfreq-neighfreq,neighwidt)
    fdatLo = fdatLo.reshape(n_channels, -1)
    fdatLo -= np.mean(fdatLo, axis=1, keepdims=True)
    covLo = np.dot(fdatLo, fdatLo.T) / n_samples

    # compute covariance matrix for upper neighbor
    fdatHi = np.zeros((n_channels, n_samples, n_trials))
    for ti in range(n_trials):
        tmdat = X[ti]
        fdatHi[:,:,ti] = filterFGx(tmdat,srate,peakfreq+neighfreq,neighwidt)
    fdatHi = fdatHi.reshape(n_channels, -1)
    fdatHi -= np.mean(fdatHi, axis=1, keepdims=True)
    covHi = np.dot(fdatHi, fdatHi.T) / n_samples

    # shrinkage regularization
    covBt = (covHi + covLo)/2
    gamma = 0.01
    evalue,_ = np.linalg.eig(covBt)
    covBt = covBt + gamma*np.mean(evalue)*np.eye(covBt.shape[0])

    evals,evecs = eigh(covAt,covBt)

    sidx  = np.argsort(evals)[::-1]
    evals = evals[sidx]
    evecs = evecs[:,sidx]

    return evecs


def _r_cca_canoncorr_withUV(X: ndarray,
                            Y: List[ndarray],
                            U: ndarray,
                            V: ndarray) -> ndarray:
    """
    Calculate correlation of CCA based on canoncorr for single trial data using existing U and V

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[ndarray]
        List of reference signals
    U : ndarray
        Spatial filter
        shape: (filterbank_num * stimulus_num * channel_num * n_component)
    V : ndarray
        Weights of harmonics
        shape: (filterbank_num * stimulus_num * harmonic_num * n_component)

    Returns
    -------
    R : ndarray
        Correlation
        shape: (filterbank_num * stimulus_num)
    """
    filterbank_num, channel_num, signal_len = X.shape
    if len(Y[0].shape)==2:
        harmonic_num = Y[0].shape[0]
    elif len(Y[0].shape)==3:
        harmonic_num = Y[0].shape[1]
    else:
        raise ValueError('Unknown data type')
    stimulus_num = len(Y)
    
    R = np.zeros((filterbank_num, stimulus_num))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        for i in range(stimulus_num):
            if len(Y[i].shape)==2:
                Y_tmp = Y[i]
            elif len(Y[i].shape)==3:
                Y_tmp = Y[i][k,:,:]
            else:
                raise ValueError('Unknown data type')
            
            A_r = U[k,i,:,:]
            B_r = V[k,i,:,:]
            
            a = A_r.T @ tmp
            b = B_r.T @ Y_tmp
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            
            # r2 = stats.pearsonr(a, b)[0]
            # r = stats.pearsonr(a, b)[0]
            r = np.corrcoef(a, b)[0,1]
            R[k,i] = r
    return R


class RESS(BaseModel):
    """"
    Rhythmic Entrainment Source Separation (RESS) method


    Reference:
    [1] Xu W., Ke Y., Ming D. Improving the Performance of Individually Calibrated SSVEP
      Classification by Rhythmic Entrainment Source Separation[J]. IEEE Transactions on 
      Neural Systems and Rehabilitation Engineering, 2024: 1-1
    """
    def __init__(self,
                 stim_freqs: List[float],
                 srate: int,
                 ress_param: dict,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None):
        super().__init__(ID = 'RESS',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.model['U'] = None # Spatial filter of EEG
        self.stim_freqs = stim_freqs
        self.srate = srate
        self.ress_param = ress_param

    def __copy__(self):
        copy_model = RESS(n_component = self.n_component,
                          stim_freqs = self.stim_freqs,
                          srate = self.srate,
                          ress_param = self.ress_param,
                          n_jobs = self.n_jobs,
                          weights_filterbank = self.model['weights_filterbank'])
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        """
        if Y is None:
            raise ValueError('TRCA requires training label')
        if X is None:
            raise ValueError('TRCA requires training data')
        
        template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        # ress parameters
        srate = self.srate
        stim_freqs = self.stim_freqs
        peakwidt = self.ress_param['peakwidt']
        neighfreq = self.ress_param['neighfreq']
        neighwidt = self.ress_param['neighwidt']

        # spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X: (filterbank_num, channel_num, signal_len)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        n_component = self.n_component
        U_trca = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
        possible_class = list(set(Y))
        possible_class.sort(reverse = False)
        for filterbank_idx in range(filterbank_num):
            X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            if self.n_jobs is not None:
                U = Parallel(n_jobs = self.n_jobs)(delayed(_ress_U)(X = X_single_class, 
                                                                    stim_freq = stim_freq,
                                                                    filterbank_idx = filterbank_idx,
                                                                    srate = srate, 
                                                                    peakwidt = peakwidt, 
                                                                    neighfreq = neighfreq, 
                                                                    neighwidt = neighwidt) for X_single_class,stim_freq in zip(X_train, stim_freqs))
            else:
                U = []
                for X_single_class,stim_freq in zip(X_train, self.stim_freqs):
                    U.append(
                        _ress_U(X = X_single_class, 
                                stim_freq = stim_freq, 
                                filterbank_idx = filterbank_idx,
                                srate = srate,
                                peakwidt = peakwidt,
                                neighfreq = neighfreq, 
                                neighwidt = neighwidt)
                    )
            for stim_idx, u in enumerate(U):
                U_trca[filterbank_idx, stim_idx, :, :] = u[:channel_num,:n_component]
        self.model['U'] = U_trca

    def predict(self,
            X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")

        template_sig = self.model['template_sig']
        U = self.model['U'] 

        if self.n_jobs is not None:
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr_withUV, Y=template_sig, U=U, V=U))(X=a) for a in X)
        else:
            r = []
            for a in X:
                r.append(
                    _r_cca_canoncorr_withUV(X=a, Y=template_sig, U=U, V=U)
                )

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]
        
        return Y_pred, r
    

class ERESS(BaseModel):
    """
    ePRCA method
    """
    def __init__(self,
                 stim_freqs: List[float],
                 srate: int,
                 ress_param: dict,
                 n_component: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None):
        if n_component is not None:
            warnings.warn("Although 'n_component' is provided, it will not considered in eTRCA")
        n_component = 1
        super().__init__(ID = 'ePRCA',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.model['U'] = None # Spatial filter of EEG
        self.stim_freqs = stim_freqs
        self.srate = srate
        self.ress_param = ress_param

    def __copy__(self):
        copy_model = ERESS(stim_freqs = self.stim_freqs,
                          srate = self.srate,
                          ress_param = self.ress_param,
                          n_component = None,
                          n_jobs = self.n_jobs,
                          weights_filterbank = self.model['weights_filterbank'])
        copy_model.model = deepcopy(self.model)
        return copy_model
    
    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        """
        template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        # ress parameters
        srate = self.srate
        stim_freqs = self.stim_freqs
        peakwidt = self.ress_param['peakwidt']
        neighfreq = self.ress_param['neighfreq']
        neighwidt = self.ress_param['neighwidt']

        # spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X: (filterbank_num, channel_num, signal_len)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]

        U_trca = np.zeros((filterbank_num, 1, channel_num, stimulus_num))
        possible_class = list(set(Y))
        possible_class.sort(reverse = False)
        for filterbank_idx in range(filterbank_num):
            X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            if self.n_jobs is not None:
                U = Parallel(n_jobs = self.n_jobs)(delayed(_ress_U)(X = X_single_class, 
                                                                    stim_freq = stim_freq,
                                                                    filterbank_idx = filterbank_idx,
                                                                    srate = srate, 
                                                                    peakwidt = peakwidt, 
                                                                    neighfreq = neighfreq, 
                                                                    neighwidt = neighwidt) for X_single_class,stim_freq in zip(X_train, stim_freqs))
            else:
                U = []
                for X_single_class,stim_freq in zip(X_train, self.stim_freqs):
                    U.append(
                        _ress_U(X = X_single_class, 
                                stim_freq = stim_freq, 
                                filterbank_idx = filterbank_idx,
                                srate = srate,
                                peakwidt = peakwidt,
                                neighfreq = neighfreq, 
                                neighwidt = neighwidt)
                    )
            for stim_idx, u in enumerate(U):
                U_trca[filterbank_idx, 0, :, stim_idx] = u[:channel_num,0]
        U_trca = np.repeat(U_trca, repeats = stimulus_num, axis = 1)

        self.model['U'] = U_trca


    def predict(self,
            X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")

        template_sig = self.model['template_sig']
        U = self.model['U'] 

        if self.n_jobs is not None:
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr_withUV, Y=template_sig, U=U, V=U))(X=a) for a in X)
        else:
            r = []
            for a in X:
                r.append(
                    _r_cca_canoncorr_withUV(X=a, Y=template_sig, U=U, V=U)
                )

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]
        
        return Y_pred, r