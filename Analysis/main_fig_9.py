# %%
%reload_ext autoreload
%autoreload 2
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib auto
import pickle
import pandas as pd


# %%
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from my_code.datasets.mymegdataset import MyMEGDataset
from my_code.datasets.myeegdataset import MyEEGDataset

import my_code.utils.megpreprocess as megpreproc
import my_code.utils.eegpreprocess as eegpreproc

projct_path = os.getcwd()

MEG = MyMEGDataset(path='datasets/OPMEEGBCI/MEG')
EEG = MyEEGDataset(path='datasets/OPMEEGBCI/EEG')


MEG.regist_preprocess(megpreproc.preprocess)
EEG.regist_preprocess(eegpreproc.preprocess)

datasets = {'MEG':MEG,'EEG':EEG}


# %%
