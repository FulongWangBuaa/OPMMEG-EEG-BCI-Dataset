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

from SSVEPAnalysisToolbox.algorithms.basemodel import BaseModel
from SSVEPAnalysisToolbox.algorithms.utils import (
    gen_template, sort, canoncorr, separate_trainSig, qr_list, blkrep, eigvec, cholesky,
    inv, repmat
)

def filterbank(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    ndim = X.ndim
    if ndim == 2:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
        axis = 1
    elif ndim == 3:
        filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1], X.shape[2]))
        axis = 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    for k in range(1, num_subbands+1, 1):
        Wp = [passband[k-1]/(srate/2), 80/(srate/2)]
        Ws = [stopband[k-1]/(srate/2), 90/(srate/2)]

        gstop = 20
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                if ndim == 2:
                    filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                else:
                    filterbank_X[k-1,:,:,:] = signal.filtfilt(bpB, bpA, X, axis = axis, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                break
            except:
                gstop -= 1
        if gstop<20:
            raise ValueError("""
                Filterbank cannot be processed. You may try longer signal lengths.
                Filterbank order: {:n}
                gstop: {:n}
                bpB: {:s}
                bpA: {:s}
                Required signal length: {:n}
                Signal length: {:n}""".format(k,
                                                gstop,
                                                str(bpB),
                                                str(bpA),
                                                3*(max(len(bpB),len(bpA))-1),
                                                X.shape[1]))
    return filterbank_X


def _prca_U(X: list, stim_freq: float, srate: float):
    """
    Calculate spatial filters of trca

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

    Ln = int(np.round(srate/stim_freq))
    Pn = int(np.floor(n_samples/Ln))
    tmp_data_prca = np.zeros((n_channels,Ln,n_trials,Pn))
    for idx_trial in range(n_trials):
        data_trial = X[idx_trial]
        for idx_pn in range(Pn):
            tmp = data_trial[:,idx_pn*Ln:(idx_pn+1)*Ln]
            tmp_data_prca[:,:,idx_trial,idx_pn] = tmp - np.mean(tmp,axis=1,keepdims=True)
    # PRCA
    # 所有周期重复成分按时间拼接
    X1 = tmp_data_prca.reshape((n_channels, -1)) # (n_channels, Ln*n_trials*Pn)
    # 所有周期重复成分求和
    X2 = np.squeeze(np.sum(tmp_data_prca, axis=(2,3))) # (n_channels, Ln)

    Q = np.dot(X1,X1.T)/X1.shape[1]
    S = np.dot(X2,X2.T)/X2.shape[1] - Q
    eig_val,eig_vec = eig(S,Q)
    sort_idx = np.argsort(eig_val)[::-1]
    U = eig_vec[:,sort_idx]

    return U

def _prca_coor2(A,B):
    '''
    CORR2 2-D correlation coefficient

    Parameters
    ----------
    A: (n_samples, n_stimulus)
    B: (n_samples, n_stimulus)

    Returns
    -------
    R: float
    '''
    if any([A.shape[0] != B.shape[0], A.shape[1] != B.shape[1]]):
        raise ValueError('A has shape {:s}, B has shape {:s}', str(A.shape), str(B.shape))
    A = np.double(A)
    B = np.double(B)

    A = A - np.mean(A)
    B = B - np.mean(B)

    R = np.sum(np.sum(A*B,axis=0,keepdims=True))/np.sqrt(np.sum(np.sum(A*A,axis=0,keepdims=True))*np.sum(np.sum(B*B,axis=0,keepdims=True)))
    return R

def _prca_gen_template(X:List[ndarray],Y: List[int],stim_freqs: List[float], srate: float):
    '''
    PRCA template calculate.

    Parameters
    ----------
    X : List[ndarray]
        Training data
        List shape: (trial_num,)
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[int]
        Training label
        List shape: (trial_num,)
    stim_freq : List[float]
        Stimulus frequency
        List shape: (stimulus_num,)
    srate : float

    Returns
    -------
    template_sig : List[ndarray]
        Template signal
        List of shape: (stimulus_num,)
        Template shape: (filterbank_num, channel_num, signal_len)
    '''
    unique_Y = list(set(Y))
    unique_Y.sort()
    template_sig = []
    for i in unique_Y:
        stim_freq = stim_freqs[i]

        # i-th class trial index
        target_idx = [k for k in range(len(Y)) if Y[k] == unique_Y[i]]
        # Get i-th class training data
        template_sig_single = [np.expand_dims(X[k], axis=0) for k in target_idx]
        template_sig_single = np.concatenate(template_sig_single, axis=0)

        n_trials, n_banks, n_channels, n_samples = template_sig_single.shape

        Ln = int(np.round(srate/stim_freq))
        Pn = int(np.floor(n_samples/Ln))

        tmp_data = [] # List shape: (trial_num*Pn)
        for idx_trial in range(n_trials):
            data_trial = template_sig_single[idx_trial,:,:,:]
            for idx_pn in range(Pn):
                tmp = data_trial[:,:,idx_pn*Ln:(idx_pn+1)*Ln]
                tmp_data.append(tmp - np.mean(tmp,axis=2,keepdims=True))

        tmp_data1 = [np.expand_dims(tmp, axis=0) for tmp in tmp_data]
        # (trial_num*Pn, n_banks, n_channels, Ln)
        tmp_data1 = np.concatenate(tmp_data1, axis=0) 

        # (n_banks, n_channels, Ln)
        tmp_data_prca = np.mean(tmp_data1,axis=0)
        
        if Ln == n_samples:
            tmplate = tmp_data_prca
        elif Ln > n_samples:
            tmplate = tmp_data_prca[:,:,:n_samples]
        else:
            n = int(np.floor(n_samples/Ln))
            tmplate = np.concatenate([tmp_data_prca]*n, axis = 2)
            tmplate = np.concatenate([tmplate,tmp_data_prca[:,:,0:n_samples-tmplate.shape[2]]],axis = 2)

        template_sig.append(tmplate)

    return template_sig

def _r_cca_canoncorr(X: ndarray,
                     Y: List[ndarray],
                     U: ndarray,
                     V: ndarray,) -> ndarray:
    """
    Calculate correlation of CCA based on canoncorr for single trial data using existing U and V

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[ndarray]
        List of reference signals
        List shape: (stimulus_num,)
        Reference shape: (filterbank_num, channel_num, signal_len)
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
    n_banks, n_channels, n_samples = X.shape
    n_stims = len(Y)

    R = np.zeros((n_banks, n_stims))

    for idx_fb in range(n_banks):
        X_tmp = X[idx_fb,:,:]
        for idx_class in range(n_stims):
            Y_tmp = Y[idx_class][idx_fb,:,:]

            w1 = U[idx_fb,idx_class,:,:]
            w2 = V[idx_fb,idx_class,:,:]

            A = X_tmp.T @ w1
            B = Y_tmp.T @ w2

            r = _prca_coor2(A,B)
            R[idx_fb,idx_class] = r
    return R

class PRCA(BaseModel):
    """
    PRCA method

    References
    ----------
    .. [1] Ke Y., Liu S., Ming D. Enhancing SSVEP Identification With Less Individual Calibration Data 
           Using Periodically Repeated Component Analysis[J]. IEEE Transactions on Biomedical Engineering, 
           2024,71(4): 1319-1331
    """
    def __init__(self,
                 stim_freqs: List[float],
                 srate: float = 1000,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None):
        super().__init__(ID = 'PRCA',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.model['U'] = None # Spatial filter of EEG
        self.srate = srate
        self.stim_freqs = stim_freqs

    def __copy__(self):
        copy_model = PRCA(stim_freqs = self.stim_freqs,
                          srate = self.srate,
                          n_component = self.n_component,
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

        # List of shape: (stimulus_num,)
        # Template shape: (filterbank_num, channel_num, signal_len)
        template_sig = _prca_gen_template(X, Y, self.stim_freqs, self.srate) 
                                          
        self.model['template_sig'] = template_sig

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
                U = Parallel(n_jobs = self.n_jobs)(delayed(_prca_U)(X=X_single_class, srate=self.srate, stim_freq=stim_freq) for X_single_class,stim_freq in zip(X_train, self.stim_freqs))
            else:
                U = []
                for X_single_class,stim_freq in zip(X_train, self.stim_freqs):
                    U.append(
                        _prca_U(X = X_single_class,srate=self.srate,stim_freq=stim_freq)
                    )
            for stim_idx, u in enumerate(U):
                U_trca[filterbank_idx, stim_idx, :, :] = u[:channel_num,:n_component]
        self.model['U'] = U_trca

    def predict(self,X: List[ndarray]) -> List[int]:
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
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr, Y=template_sig, U=U, V=U))(X=a) for a in X)
        else:
            r = []
            for a in X:
                r.append(_r_cca_canoncorr(X=a, Y=template_sig, U=U, V=U))

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]

        return Y_pred, r


class EPRCA(BaseModel):
    """
    ePRCA method
    """
    def __init__(self,
                 stim_freqs: List[float],
                 srate: float = 1000,
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
        self.srate = srate
        self.stim_freqs = stim_freqs

    def __copy__(self):
        copy_model = EPRCA(stim_freqs = self.stim_freqs,
                          srate = self.srate,
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
        if Y is None:
            raise ValueError('eTRCA requires training label')
        if X is None:
            raise ValueError('eTRCA requires training data')

        template_sig = _prca_gen_template(X, Y, self.stim_freqs, self.srate)  # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        # spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X: (filterbank_num, channel_num, signal_len)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        # n_component = 1
        U_trca = np.zeros((filterbank_num, 1, channel_num, stimulus_num))
        possible_class = list(set(Y))
        possible_class.sort(reverse = False)
        for filterbank_idx in range(filterbank_num):
            X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            if self.n_jobs is not None:
                U = Parallel(n_jobs = self.n_jobs)(delayed(_prca_U)(X=X_single_class, srate=self.srate, stim_freq=stim_freq) for X_single_class,stim_freq in zip(X_train, self.stim_freqs))
            else:
                U = []
                for X_single_class,stim_freq in zip(X_train, self.stim_freqs):
                    U.append(
                        _prca_U(X = X_single_class,srate=self.srate,stim_freq=stim_freq)
                    )
            # U = []
            # for X_single_class in X_train:
            #     U_element = _trca_U(X = X_single_class)
            #     U.append(U_element)
            for stim_idx, u in enumerate(U):
                U_trca[filterbank_idx, 0, :, stim_idx] = u[:channel_num,0]
        U_trca = np.repeat(U_trca, repeats = stimulus_num, axis = 1)

        self.model['U'] = U_trca


    def predict(self,X: List[ndarray]) -> List[int]:
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
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr, Y=template_sig, U=U, V=U))(X=a) for a in X)
        else:
            r = []
            for a in X:
                r.append(_r_cca_canoncorr(X=a, Y=template_sig, U=U, V=U))

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]

        return Y_pred, r


def _trcaR_cal_template_U(X_single_stimulus : ndarray,
                          I : ndarray,
                          n_component : int):
    """
    Calculate templates and trials' spatial filters in TRCA-R
    """
    trial_num, filterbank_num, channel_num, signal_len = X_single_stimulus.shape
    # prepare center matrix
    # I = np.eye(signal_len)
    LL = repmat(I, trial_num, trial_num) - blkrep(I, trial_num)
    # calculate spatial filters of each filterbank
    U_trial = []
    for filterbank_idx in range(filterbank_num):
        X_single_stimulus_single_filterbank = X_single_stimulus[:,filterbank_idx,:,:]
        template = []
        for trial_idx in range(trial_num):
            template.append(X_single_stimulus_single_filterbank[trial_idx,:,:])
        template = np.concatenate(template, axis = 1)
        # calculate spatial filters of trials
        try:
            Sb = template @ LL @ template.T
        except:
            raise ValueError(template.shape, LL.shape, signal_len,I.shape,trial_num)
        Sw = template @ template.T
        eig_vec = eigvec(Sb, Sw)[:channel_num,:n_component]
        U_trial.append(np.expand_dims(eig_vec, axis = 0))
    U_trial = np.concatenate(U_trial, axis = 0)
    return U_trial


class PRCAwithR(BaseModel):
    """
    PRCA method with reference signals (PRCA-R)

    References
    ----------
    .. [1] Ke Y., Liu S., Ming D. Enhancing SSVEP Identification With Less Individual Calibration Data 
           Using Periodically Repeated Component Analysis[J]. IEEE Transactions on Biomedical Engineering, 
           2024,71(4): 1319-1331

    tips: 
    ----------
    In PRCA paper, the author replace the multi-trial average template with 
    the PRC-based synthetic template in correlation-based feature calculation 
    without changing the calculation method of the spatial filters in the method 
    using TRCA-R with PRC-based synthetic templates (PRCA-R).
    """
    def __init__(self,
                 stim_freqs: List[float],
                 srate: float = 1000,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None):
        super().__init__(ID = 'PRCA-R',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.model['U'] = None # Spatial filter of EEG
        self.stim_freqs = stim_freqs
        self.srate = srate

    def __copy__(self):
        copy_model = PRCAwithR(stim_freqs=self.stim_freqs,
                               srate=self.srate,
                               n_component = self.n_component,
                               n_jobs = self.n_jobs,
                               weights_filterbank = self.model['weights_filterbank'])
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None,
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
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
            Reference signal shape: (harmonic_num, signal_len)
        """
        if Y is None:
            raise ValueError('TRCA with reference signals requires training label')
        if X is None:
            raise ValueError('TRCA with reference signals training data')
        if ref_sig is None:
            raise ValueError('TRCA with reference signals requires sine-cosine-based reference signal')

        template_sig = _prca_gen_template(X, Y, self.stim_freqs, self.srate) # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        # Template signal
        # List of shape: (stimulus_num,)
        # Template shape: (trial_num, filterbank_num, channel_num, signal_len)
        separated_trainSig = separate_trainSig(X, Y)

        ref_sig_Q, ref_sig_R, ref_sig_P = qr_list(ref_sig)

        if self.n_jobs is not None:
            U_all_stimuli = Parallel(n_jobs=self.n_jobs)(delayed(partial(_trcaR_cal_template_U, n_component = self.n_component))(X_single_stimulus = a, I = Q @ Q.T) for a, Q in zip(separated_trainSig, ref_sig_Q))
        else:
            U_all_stimuli = []
            for a, Q in zip(separated_trainSig, ref_sig_Q):
                U_all_stimuli.append(
                    _trcaR_cal_template_U(X_single_stimulus = a, I = Q @ Q.T, n_component = self.n_component)
                )
        U_trca = [np.expand_dims(u, axis=1) for u in U_all_stimuli]
        U_trca = np.concatenate(U_trca, axis = 1)
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
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr, Y=template_sig, U=U, V=U))(X=a) for a in X)
        else:
            r = []
            for a in X:
                r.append(
                    _r_cca_canoncorr(X=a, Y=template_sig, U=U, V=U)
                )

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]

        return Y_pred, r


# class EPRCAwithR(BaseModel):
#     """
#     ePRCA method with reference signals (eTRCA-R)
#     """
#     def __init__(self,
#                  stim_freqs: List[float],
#                  srate: float = 1000,
#                  n_component: Optional[int] = None,
#                  n_jobs: Optional[int] = None,
#                  weights_filterbank: Optional[List[float]] = None):
#         if n_component is not None:
#             warnings.warn("Although 'n_component' is provided, it will not considered in eTRCA")
#         n_component = 1
#         super().__init__(ID = 'ePRCA-R',
#                          n_component = n_component,
#                          n_jobs = n_jobs,
#                          weights_filterbank = weights_filterbank)
#         self.model['U'] = None # Spatial filter of EEG

#     def __copy__(self):
#         copy_model = EPRCAwithR(n_component = None,
#                                 n_jobs = self.n_jobs,
#                                 weights_filterbank = self.model['weights_filterbank'])
#         copy_model.model = deepcopy(self.model)
#         return copy_model

#     def fit(self,
#             X: Optional[List[ndarray]] = None,
#             Y: Optional[List[int]] = None,
#             ref_sig: Optional[List[ndarray]] = None,
#             *argv, **kwargs):
#         """
#         Parameters
#         -------------
#         X : Optional[List[ndarray]], optional
#             List of training EEG data. The default is None.
#             List shape: (trial_num,)
#             EEG shape: (filterbank_num, channel_num, signal_len)
#         Y : Optional[List[int]], optional
#             List of labels (stimulus indices). The default is None.
#             List shape: (trial_num,)
#         ref_sig : Optional[List[ndarray]], optional
#             Sine-cosine-based reference signals. The default is None.
#             List of shape: (stimulus_num,)
#             Reference signal shape: (harmonic_num, signal_len)
#         """
#         if Y is None:
#             raise ValueError('eTRCA with reference signals requires training label')
#         if X is None:
#             raise ValueError('eTRCA with reference signals training data')
#         if ref_sig is None:
#             raise ValueError('eTRCA with reference signals requires sine-cosine-based reference signal')

#         template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
#                                           # Template shape: (filterbank_num, channel_num, signal_len)
#         self.model['template_sig'] = template_sig

#         separated_trainSig = separate_trainSig(X, Y)
#         ref_sig_Q, ref_sig_R, ref_sig_P = qr_list(ref_sig)

#         if self.n_jobs is not None:
#             U_all_stimuli = Parallel(n_jobs=self.n_jobs)(delayed(partial(_trcaR_cal_template_U, n_component = self.n_component))(X_single_stimulus = a, I = Q @ Q.T) for a, Q in zip(separated_trainSig, ref_sig_Q))
#         else:
#             U_all_stimuli = []
#             for a, Q in zip(separated_trainSig, ref_sig_Q):
#                 U_all_stimuli.append(
#                     _trcaR_cal_template_U(X_single_stimulus = a, I = Q @ Q.T, n_component = self.n_component)
#                 )
#         # U_trca = [u for u in U_all_stimuli]
#         U_trca = np.concatenate(U_all_stimuli, axis = 2)
#         U_trca = np.expand_dims(U_trca, axis = 1)
#         U_trca = np.repeat(U_trca, repeats = len(U_all_stimuli), axis = 1)
#         self.model['U'] = U_trca

#     def predict(self,
#             X: List[ndarray]) -> List[int]:
#         weights_filterbank = self.model['weights_filterbank']
#         if weights_filterbank is None:
#             weights_filterbank = [1 for _ in range(X[0].shape[0])]
#         if type(weights_filterbank) is list:
#             weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
#         else:
#             if len(weights_filterbank.shape) != 2:
#                 raise ValueError("'weights_filterbank' has wrong shape")
#             if weights_filterbank.shape[0] != 1:
#                 weights_filterbank = weights_filterbank.T
#         if weights_filterbank.shape[0] != 1:
#             raise ValueError("'weights_filterbank' has wrong shape")

#         template_sig = self.model['template_sig']
#         U = self.model['U'] 

#         if self.n_jobs is not None:
#             r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr_withUV, Y=template_sig, U=U, V=U))(X=a) for a in X)
#         else:
#             r = []
#             for a in X:
#                 r.append(
#                     _r_cca_canoncorr_withUV(X=a, Y=template_sig, U=U, V=U)
#                 )

#         Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]
        
#         return Y_pred, r
