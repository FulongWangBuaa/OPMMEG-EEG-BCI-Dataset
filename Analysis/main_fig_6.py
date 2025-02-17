# %%
%reload_ext autoreload
%autoreload 2
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

from my_code.utils.meegutils import find_nearest_index,amplitude_spectrum

projct_path = os.getcwd()

MEG = MyMEGDataset(path='datasets/OPMEEGBCI/MEG')
EEG = MyEEGDataset(path='datasets/OPMEEGBCI/EEG')


MEG.regist_preprocess(megpreproc.preprocess)
EEG.regist_preprocess(eegpreproc.preprocess)

datasets = {'MEG':MEG,'EEG':EEG}


# %%
sfreq = MEG.srate
stim_id = MEG.stim_info['stim_id']
stim_freqs = MEG.stim_info['freqs']
stim_phases = MEG.stim_info['phases']
stim_dict = dict()
for i in range(len(stim_id)):
    stim_dict[str(stim_freqs[i])+'Hz'] = stim_id[i]


# %%----------------------------------------fig 6----------------------------------------
info_meg = datasets['MEG'].get_sub_info(sub_idx=0)
info_eeg = datasets['EEG'].get_sub_info(sub_idx=0)

spectrum = dict()
for dataset_key in datasets.keys():
    spectrum[dataset_key] = dict()
    dataset = datasets[dataset_key]
    dataset_data = np.array(dataset.get_all_data())
    dataset_data_group = np.mean(dataset_data,axis=1) # sub_num, stim_num, ch_num, sample_num

    spectrum_subs = []
    for sub_idx in range(dataset.sub_num):
        dataset_data_sub = dataset_data_group[sub_idx] # stim_num, ch_num, sample_num
        spectrum_stims = []
        for stim_idx in range(len(stim_freqs)):
            dataset_data_stim = dataset_data_sub[stim_idx,:,:]
            f, spectrum_stim = amplitude_spectrum(dataset,dataset_data_stim)
            spectrum_stims.append(spectrum_stim)

        spectrum_stims = np.array(spectrum_stims)
        spectrum_subs.append(spectrum_stims)
    spectrum_subs = np.array(spectrum_subs)

    spectrum[dataset_key]['spectrum'] = np.mean(spectrum_subs,axis=0)
    spectrum[dataset_key]['f'] = f

pick_chs = ['O2','O1','POZ','OZ']
ch_meg_idx = [datasets['MEG'].channels.index(ch) for ch in pick_chs]
ch_eeg_idx = [datasets['EEG'].channels.index(ch) for ch in pick_chs]

pick_stim = 11
pick_stim_idx = stim_freqs.index(pick_stim)

spectrum_mean_meg = np.mean(spectrum['MEG']['spectrum'][pick_stim_idx,ch_meg_idx,:],axis=0)
spectrum_mean_eeg = np.mean(spectrum['EEG']['spectrum'][pick_stim_idx,ch_eeg_idx,:],axis=0)


# %%
eeg_color = '#4682B4'
meg_color = '#CD5C5C'

colors = ['#026CCBFF', '#F51E02FF', '#05B102FF', 
            '#40E0D0', '#9B9B9BFF', '#FF00FF', '#A0522D', '#FF8C00']

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'normal'

fig = plt.figure(figsize=(8, 4.5))
gs = gridspec.GridSpec(1, 1,width_ratios=[1], height_ratios=[1])
axes = fig.add_subplot(gs[0,0])
axes.axis('off')

ax = axes.inset_axes([0,0.8,1,0.3])
ax.axis('off')

vms = [(-10,60),(-10,50),(-5,15),(-5,12),(-5,10),(-2,5),(-2,5),(-1,4)]
for nh,vm in zip(range(8),vms):
    axx = ax.inset_axes([0.128*nh-0.05,0.1,0.2,0.8])
    
    freq_idx = find_nearest_index(spectrum['MEG']['f'], pick_stim*(nh+1))
    im,_ = mne.viz.plot_topomap(spectrum['MEG']['spectrum'][2,:,freq_idx],info_meg,axes=axx,
                                    contours=5,sphere=0.125,
                                    vlim=vm,cmap='RdBu_r',extrapolate='head')

ax = axes.inset_axes([0,0.55,1,0.25])
ax.plot(spectrum['MEG']['f'], spectrum_mean_meg,
        linewidth=1.2,color='k')
f_pick_idx = find_nearest_index(spectrum['MEG']['f'], [pick_stim * j for j in range(1,9)])
ax.scatter([pick_stim * j for j in range(1,9)], spectrum_mean_meg[f_pick_idx],
        s=15,c=None,marker='o',facecolors='none',edgecolors=colors)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.set_xlim([7,93])
ax.set_xticks([])
ax.set_ylim([0,50])
ax.set_yticks([0,10,20,30,40,50])
ax.set_ylabel(f'Amplitude (fT)',fontweight='normal')





ax = axes.inset_axes([0,0.25,1,0.3])
ax.axis('off')

vms = [(0,0.9),(0,0.8),(0,0.2),(0,0.2),(0,0.1),(0,0.1),(0,0.1),(0,0.1)]
for nh,vm in zip(range(8),vms):
    axx = ax.inset_axes([0.128*nh-0.05,0.1,0.2,0.8])
    freq_idx = find_nearest_index(spectrum['EEG']['f'], pick_stim*(nh+1))
    im,_ = mne.viz.plot_topomap(spectrum['EEG']['spectrum'][2,:,freq_idx],info_eeg,axes=axx,
                                    contours=5,sphere=0.1,
                                    vlim=vm,cmap='RdBu_r',extrapolate='head')

ax = axes.inset_axes([0,0,1,0.25])
ax.plot(spectrum['EEG']['f'], spectrum_mean_eeg,
        linewidth=1.2,color='k')
f_pick_idx = find_nearest_index(spectrum['EEG']['f'], [pick_stim * j for j in range(1,9)])
ax.scatter([pick_stim * j for j in range(1,9)], spectrum_mean_eeg[f_pick_idx],
        s=15,c=None,marker='o',facecolors='none',edgecolors=colors)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim([7,93])
ax.set_xticks([10,20,30,40,50,60,70,80,90])
ax.set_ylim([0,0.8])
ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax.set_xlabel('Frequency (Hz)',fontweight='normal')
ax.set_ylabel(f'Amplitude (uV)',fontweight='normal')


fig.text(0.13,0.95,'11Hz',ha='center',va='center')
fig.text(0.242,0.95,'22Hz',ha='center',va='center')
fig.text(0.356,0.95,'33Hz',ha='center',va='center')
fig.text(0.47,0.95,'44Hz',ha='center',va='center')
fig.text(0.583,0.95,'55Hz',ha='center',va='center')
fig.text(0.694,0.95,'66Hz',ha='center',va='center')
fig.text(0.811,0.95,'77Hz',ha='center',va='center')
fig.text(0.925,0.95,'88Hz',ha='center',va='center')

fig.text(0,0.92,'A',fontweight='bold',fontsize=16)
fig.text(0,0.48,'B',fontweight='bold',fontsize=16)

plt.tight_layout()
plt.show()

# %%
