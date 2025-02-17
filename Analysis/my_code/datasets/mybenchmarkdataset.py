import numpy as np
from typing import Union, Optional, Dict, List, Tuple
from SSVEPAnalysisToolbox.datasets.benchmarkdataset import BenchmarkDataset

class MyBenchmarkDataset(BenchmarkDataset):
    """My custom benchmark dataset class.

    This class inherits from the BenchmarkDataset class and can be used to create a custom benchmark dataset.
    """

    def __init__(self,
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None):
        super().__init__(path = path,
                         path_support_file = path_support_file)
        self.ch_num = len(self._CHANNELS)
        self.sub_num = len(self._SUBJECTS)

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

    def get_all_data(self):
        sub_num = self.sub_num
        data = []
        for sub_idx in range(sub_num):
            sub_data = self.get_sub_data(sub_idx)
            data.append(sub_data)
        return data