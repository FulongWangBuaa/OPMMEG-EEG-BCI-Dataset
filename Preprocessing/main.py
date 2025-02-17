# %%
%reload_ext autoreload
%autoreload 2
import sys
import scipy.io as scio
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
%matplotlib auto
import pandas as pd
import random
import os
import os.path as op
from pathlib import Path
import heapq
import copy

# MNE
import mne
from mne.io import curry
from mne import transforms
from mne.simulation import simulate_raw,simulate_sparse_stc
from mne.viz.backends.renderer import _get_renderer
from mne.transforms import apply_trans,_get_trans,_get_transforms_to_coord_frame,_frame_to_str
from mne.io.pick import _picks_to_idx

# My
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from my_code.utils import read_eeg_data, read_meg_data

# %%
project_path = os.getcwd()
sets_type = ['EEG','OPM-MEG']

raws = dict()
for set_type in sets_type:
    set_data_path = op.join(project_path,'Raw data')
    subject_list = os.listdir(set_data_path)
    for subject in subject_list:
        if set_type == 'EEG':
            subject_data_name = subject + '_eeg-raw.fif'
            subject_trigger_name = subject + '_eeg_event.txt'
        elif set_type == 'OPM-MEG':
            subject_data_name = subject + '_opmmeg-raw.fif'
            subject_events_name = subject + '_opmmeg_event.txt'
        subject_data_path = op.join(set_data_path,subject,set_type,subject_data_name)
        events_path = op.join(set_data_path,subject,set_type,subject_events_name)

        if subject not in raws.keys():
            raws[subject] = dict()

        raw = mne.io.read_raw_fif(subject_data_path,preload=True)
        event = mne.read_events(events_path)
        
        raws[subject][set_type + '_raw'] = raw
        raws[subject][set_type + '_events'] = events


# %%
event_dict = {'9Hz':173,'10Hz':178,'11Hz':183,
              '12Hz':188,'13Hz':193,'14Hz':198,
              '15Hz':203,'16Hz':208,'17Hz':213}

epochs_all = dict()

# EEG预处理
for subject in raws.keys():
    if 'EEG_raw' in raws[subject].keys():
        raw_eeg = raws[subject]['EEG_raw'].copy()

        events_eeg = raws[subject]['EEG_events']

        # 1. EEG平均参考
        raw_eeg_ref = raw_eeg.copy().set_eeg_reference(ref_channels='average')

        # 2. 去工频干扰
        raw_eeg_filter = raw_eeg_ref.copy().filter(l_freq=6,h_freq=90,fir_design="firwin",verbose=False)
        from wfl_preproc_spectrum_interpolation import spectrum_interpolation
        Fl = [50]
        dftbandwidth = [2]
        dftneighbourwidth = [2]
        raw_interpolation = spectrum_interpolation(raw_eeg_filter.copy(),Fl, dftbandwidth, dftneighbourwidth)

        # 3. Epochs
        epochs_eeg = mne.Epochs(raw_interpolation,events_eeg,event_dict,tmin=-0.5,tmax=5.5,
                                baseline=(-0.5,0),preload=True)

        epochs_all[subject] = dict()
        epochs_all[subject]['EEG_epochs'] = epochs_eeg

# %%
# MEG预处理
from my_code.utils import read_room_data
data_room_path = op.join(project_path,'support_files','20241016 220745_wfl_kfj.basedata')
sensor_path = op.join(project_path,'support_files',sensor_name)
raw_room = read_room_data(data_room_path,sensor_path)

event_dict = {'9Hz':173,'10Hz':178,'11Hz':183,
              '12Hz':188,'13Hz':193,'14Hz':198,
              '15Hz':203,'16Hz':208,'17Hz':213}

for subject in raws.keys():
    # if subject in ['whl','wfl6','wfl7']:
    if 'MEG_raw' in raws[subject].keys():
        raw_meg = raws[subject]['MEG_raw'].copy()
        events_meg = raws[subject]['MEG_events']

        # raw_meg.info['bads'] = ['CP4','P9','CP3','F7','IZ','F1','C2','C5','CZ']

        # 1. 去工频干扰
        raw_meg_filter = raw_meg.copy().filter(l_freq=6,h_freq=90,fir_design="firwin",verbose=False)

        raw_hfc = raw_meg_filter.copy()
        projs = mne.preprocessing.compute_proj_hfc(raw_hfc.info, order=2)
        raw_hfc.add_proj(projs).apply_proj()

        from wfl_preproc_spectrum_interpolation import spectrum_interpolation
        Fl = [31,50,63]
        dftbandwidth = [2,2,2]
        dftneighbourwidth = [2,2,2]
        raw_interpolation = spectrum_interpolation(raw_hfc.copy(),Fl, dftbandwidth, dftneighbourwidth)

        # 5. Epochs
        epochs_meg = mne.Epochs(raw_interpolation,events_meg,event_dict,tmin=-0.5,tmax=5.5,
                                baseline=(-0.5,0),preload=True)

        if subject not in epochs_all.keys():
            epochs_all[subject] = dict()
        epochs_all[subject]['MEG_epochs'] = epochs_meg

# %%
evoked_eeg = epochs_all['wfl']['EEG_epochs'].average()
evoked_meg = epochs_all['wfl']['MEG_epochs'].average()

evoked_eeg.plot(xlim=(-0.1,0.5))
plt.show()

evoked_meg.plot(xlim=(-0.1,0.5),exclude=['CP4','P9','CP3'])
plt.show()


# %%
raw_meg_filter.compute_psd(fmin=0,fmax=80).plot()
plt.show()

raw_interpolation.compute_psd(fmin=0,fmax=80).plot()
plt.show()



# %%
def get_epochs_data(epochs):
    num_trials,num_chans,num_samples = epochs.get_data(picks='data',copy=True).shape
    events_id = epochs.event_id
    num_stims = len(events_id)
    num_blocks = num_trials//num_stims
    data = np.zeros((num_stims,15,num_chans,num_samples))
    for stim_idx,key in enumerate(events_id.keys()):
        epochs_pick = epochs[key]
        epochs_pick_data = epochs_pick.get_data(picks='data',copy=True)
        data[stim_idx,:,:,:] = epochs_pick_data
    data = np.transpose(data, (2,3,0,1))
    return data

# %%
if True:
    for i,subject in enumerate(epochs_all.keys()):
        if 'EEG_epochs' in epochs_all[subject].keys():
            print(subject)
            epochs_eeg = epochs_all[subject]['EEG_epochs']
            data_eeg = get_epochs_data(epochs_eeg)
            epochs_eeg.info.save(f'S{i+1}-info.fif')
            scio.savemat(f'S{i+1}.mat', {'data': data_eeg})

# %%
if True:
    i = 0
    for subject in epochs_all.keys():
        if 'MEG_epochs' in epochs_all[subject].keys():
            print(subject)
            epochs_meg = epochs_all[subject]['MEG_epochs']
            data_meg = get_epochs_data(epochs_meg)
            epochs_meg.info.save(f'S{i+1}-info.fif')
            scio.savemat(f'S{i+1}.mat', {'data': data_meg})
            i = i+1




# %%
