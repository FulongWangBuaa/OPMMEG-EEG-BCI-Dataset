import os
import numpy as np
import py7zr
import mne
import scipy.io as sio

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, transpose

from SSVEPAnalysisToolbox.datasets.basedataset import BaseDataset
from SSVEPAnalysisToolbox.datasets.subjectinfo import SubInfo
from SSVEPAnalysisToolbox.utils.download import download_single_file
from SSVEPAnalysisToolbox.utils.io import loadmat

class MyEEGDataset(BaseDataset):
    """EEG Dataset for SSVEP data from my own experiments.

    Parameters
    ----------
    data_path : str
        Path to the dataset.
    subject_info : SubInfo
        Subject information.
    """

    _unit = 'uV'
    _mri_path = "D:\科研\MRI"
    _mri_name = "wangfulong1"

    _CHANNELS = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3',
                 'F1','FZ','F2','F4','F6','F8','FC5', 'FC3',
                 'FC1','FCZ','FC2','FC4','FC6','T7','C5',
                 'C3','C1','CZ','C2','C4','C6','T8','TP7','CP5',
                 'CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7',
                 'P5','P3','P1','PZ','P2','P4','P6','P8','PO7',
                 'PO3','POZ','PO4','PO8','O1','OZ','O2']

    _FREQS = [9,10,11,12,13,14,15,16,17]
    _PHASES = [0, 1, 0, 1.5, 0.5, 1.5, 0, 1, 0]

    _NAME = ['cfz','gjw','lwj','smx','ss','wcx','wfl','xch','yjx','yjz','ynh','yzb','zzp']
    _AGE = [29,24,23,27,23,24,25,23,30,23,21,23,24]
    _GENDER = ['male','male','female','male','male','male','male','male','male',
                'male','male','male','male']

    _SUBJECTS = [SubInfo(ID='S{:d}'.format(sub_idx),name=sname) for sub_idx,sname,sage,sgender in zip(range(1,len(_NAME)+1,1),_NAME,_AGE,_GENDER)]

    def __init__(self,
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None):
        super().__init__(subjects = self._SUBJECTS, 
                         ID = 'EEG',
                         url = '',
                         paths = path,
                         channels = self._CHANNELS,
                         srate = 1000,
                         block_num = 15,
                         trial_num = len(self._FREQS),
                         trial_len = 6,
                         stim_info = {'stim_num': len(self._FREQS),
                                      'stim_id':[173,178,183,188,193,198,203,208,213],
                                      'freqs': self._FREQS,
                                      'phases': [i * np.pi for i in self._PHASES]},
                         support_files = [],
                         path_support_file = path_support_file,
                         t_prestim = 0.5,
                         t_break = 0.5,
                         default_t_latency = 0.14)
        
        self.ch_num = len(self._CHANNELS)
        self.sub_num = len(self._SUBJECTS)
        self.mri_path = self._mri_path
        self.mri_name = self._mri_name
        self.unit = self._unit

        self.trial_label_check_list = {}
        for trial_i in range(self.trial_num):
            self.trial_label_check_list[trial_i] = trial_i

    def download_single_subject(self,
                                subject: SubInfo):
        source_url = self.url + subject.ID + '.mat.7z'
        desertation = os.path.join(subject.path, subject.ID + '.mat.7z')
        
        data_file = os.path.join(subject.path, subject.ID + '.mat')

        download_flag = True
        
        if not os.path.isfile(data_file):
            try:
                download_single_file(source_url, desertation)
            
                with py7zr.SevenZipFile(desertation,'r') as archive:
                    archive.extractall(subject.path)
                    
                os.remove(desertation)
            except:
                download_flag = False
        
        return download_flag, source_url, desertation
    
    def download_file(self,
                      file_name: str):
        source_url = self.url + file_name
        desertation = os.path.join(self.path_support_file, file_name)

        download_flag = True
        
        if not os.path.isfile(desertation):
            try:
                download_single_file(source_url, desertation)
            except:
                download_flag = False

        return download_flag, source_url, desertation

    def get_sub_data(self, 
                     sub_idx: int) -> ndarray:
        if sub_idx < 0:
            raise ValueError('Subject index cannot be negative')
        if sub_idx > len(self.subjects)-1:
            raise ValueError('Subject index should be smaller than {:d}'.format(len(self.subjects)))
        
        sub_info = self.subjects[sub_idx]
        file_path = os.path.join(sub_info.path, sub_info.ID + '.mat')

        try:
            mat_data = loadmat(file_path)
        except:
            mat_data = sio.loadmat(file_path)
        data = mat_data['data']
        data = transpose(data, (3,2,0,1)) # block_num * stimulus_num * ch_num * whole_trial_samples
        data = data[0:self.block_num,:,:,:]
        # if self.unit == 'V':
        data = data * 1e6
        return data
    
    def get_all_data(self):
        sub_num = self.sub_num
        data = []
        for sub_idx in range(sub_num):
            sub_data = self.get_sub_data(sub_idx)
            data.append(sub_data)
        return data

    def get_sub_info(self,
                     sub_idx: int):
        if sub_idx < 0:
            raise ValueError('Subject index cannot be negative')
        if sub_idx > len(self.subjects)-1:
            raise ValueError('Subject index should be smaller than {:d}'.format(len(self.subjects)))

        sub_info = self.subjects[sub_idx]
        file_path = os.path.join(sub_info.path, sub_info.ID + '-info.fif')

        info = mne.io.read_info(file_path,verbose=False)
        return info

    def get_data_single_channel(self,chs):
        sub_num = self.sub_num
        trial_num = self.trial_num
        block_num = self.block_num
        ch_num = self.ch_num

        if not isinstance(chs,list):
            chs = [chs]
        # subj * block_num * stimulus_num * ch_num
        data = []
        sub_idxs = np.arange(sub_num)
        for sub_idx in sub_idxs:
            sub_data = self.get_sub_data(sub_idx)
            sample_num = sub_data.shape[3]
            sub_data_pick = np.zeros((block_num, trial_num, len(chs),sample_num))
            for idx,ch in enumerate(chs):
                if isinstance(ch, str):
                    ch_idx = self.get_ch_idx(ch)
                elif isinstance(ch, int):
                    ch_idx = ch
                sub_data_pick[:,:,idx,:] = sub_data[:,:,ch_idx,:]
            data.append(sub_data_pick)
        return data


    def get_label_single_trial(self,
                               sub_idx: int,
                               block_idx: int,
                               trial_idx: int) -> int:
        return self.trial_label_check_list[trial_idx]