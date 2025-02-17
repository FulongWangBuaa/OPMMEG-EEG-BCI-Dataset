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


projct_path = os.getcwd()

MEG = MyMEGDataset(path='datasets/OPMEEGBCI/MEG')
EEG = MyEEGDataset(path='datasets/OPMEEGBCI/EEG')


MEG.regist_preprocess(megpreproc.preprocess)
EEG.regist_preprocess(eegpreproc.preprocess)

datasets = {'MEG':MEG,'EEG':EEG}


# %%----------------------------------------fig 7A----------------------------------------
# from SSVEPAnalysisToolbox.evaluator.baseevaluator import create_pbar
# from SSVEPAnalysisToolbox.utils.algsupport import freqs_snr
# from SSVEPAnalysisToolbox.utils.algsupport import nextpow2
# harmonic_num = 5
# sig_len = 3
# snrs_fft = dict()
# for dataset_key in ['MEG','EEG']:
#     dataset = datasets[dataset_key]
#     sub_num = dataset.sub_num

#     ch_num = dataset.ch_num
#     block_num = dataset.block_num
#     trial_num = dataset.trial_num

#     dataset_data = dataset.get_all_data()
#     idx_star = int(dataset.t_prestim*dataset.srate)
#     idx_end = int(dataset.t_prestim * dataset.srate + sig_len * dataset.srate)
#     data = [i[:,:,:,idx_star:idx_end] for i in dataset_data]

#     pbar = create_pbar([sub_num, block_num])
#     snr = np.zeros((sub_num, block_num, trial_num, ch_num))

#     for sub_idx in range(sub_num):
#         for block_idx in range(block_num):
#             pbar.update(1)
#             for trial_idx in range(trial_num):
#                 for ch_idx in range(ch_num):
#                     X_fft = data[sub_idx][block_idx, trial_idx, ch_idx,:]
#                     X_fft = bandpass_filter(dataset, X_fft, 6, 90)
#                     X_fft = preprocess(dataset, X_fft)
#                     X_fft = X_fft[np.newaxis, :]
#                     snr[sub_idx, block_idx, trial_idx, ch_idx] = freqs_snr(X_fft, dataset.stim_info['freqs'][trial_idx], dataset.srate, Nh = harmonic_num,
#                                                                             NFFT = 2 ** nextpow2(10*dataset.srate))
#     snrs_fft[dataset_key] = snr
#     del pbar

with open(r'./results/snrs_fft.pkl','rb') as f:
    snrs_fft = pickle.load(f)

# 每个刺激对应的信噪比
picks_meg = ['PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']
picks_eeg = ['PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']


picks_meg_idx = [datasets['MEG'].channels.index(pick_ch) for pick_ch in picks_meg]
picks_eeg_idx = [datasets['EEG'].channels.index(pick_ch) for pick_ch in picks_eeg]


snr_fft_meg_pick = snrs_fft['MEG'][:,:,:,picks_meg_idx]
snr_fft_eeg_pick = snrs_fft['EEG'][:,:,:,picks_eeg_idx]

snr_fft_meg_pick = np.reshape(np.transpose(snr_fft_meg_pick,(2,0,1,3)),(9,-1))
snr_fft_eeg_pick = np.reshape(np.transpose(snr_fft_eeg_pick,(2,0,1,3)),(9,-1))

snr_fft_meg_pick = [snr_fft_meg_pick[i] for i in range(snr_fft_meg_pick.shape[0])]
snr_fft_eeg_pick = [snr_fft_eeg_pick[i] for i in range(snr_fft_eeg_pick.shape[0])]

import matplotlib.pyplot as plt
import numpy as np

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')




plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(3, 4.4),constrained_layout=True)


parts = ax1.violinplot(
        snr_fft_meg_pick, showmeans=False, showmedians=False,
        showextrema=False)

from pypalettes import load_cmap
cmap = load_cmap("BluetoOrange_10")
colors = cmap.colors
for pc,color in zip(parts['bodies'],colors[1:]):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(snr_fft_meg_pick, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(snr_fft_meg_pick, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax1.scatter(inds, medians, marker='o', color='white', s=5, zorder=3)
ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax1.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax1.set_ylim(-31,-14)
ax1.set_yticks([-30,-25,-20,-15])
ax1.set_xticks([1,2,3,4,5,6,7,8,9])
ax1.set_xticklabels(['9','10','11','12','13','14','15','16','17'])

ax1.set_ylabel('SNR(dB)')
ax1.set_xlabel('Stimulus Frequency (Hz)')
ax1.set_title('(a) OPM-MEG',fontsize=10)



parts = ax2.violinplot(
        snr_fft_eeg_pick, showmeans=False, showmedians=False,
        showextrema=False)

from pypalettes import load_cmap
cmap = load_cmap("Hiroshige")
colors = cmap.colors
for pc,color in zip(parts['bodies'],colors[1:]):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(snr_fft_eeg_pick, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(snr_fft_eeg_pick, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=5, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.set_ylim(-31,-14)
ax2.set_yticks([-30,-25,-20,-15])
ax2.set_xticks([1,2,3,4,5,6,7,8,9])
ax2.set_xticklabels(['9','10','11','12','13','14','15','16','17'])

ax2.set_ylabel('SNR(dB)')
ax2.set_xlabel('Stimulus Frequency (Hz)')
ax2.set_title('(b) EEG',fontsize=10)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()


# fig.savefig('snr_stim.png',dpi=600)
# fig.savefig('snr_stim.pdf')
# fig.savefig('snr_stim.svg')


# %%----------------------------------------fig 7B----------------------------------------
# from SSVEPAnalysisToolbox.utils.algsupport import nextpow2
# phases = dict()
# for dataset_key in ['MEG','EEG']:
#     dataset = datasets[dataset_key]
#     phase = dataset.get_phase(display_progress = True,
#                         sig_len = 4,
#                         remove_break = False,
#                         remove_pre_and_latency = False,
#                         remove_target_phase = False,
#                         NFFT = 2 ** nextpow2(10*dataset.srate))

#     phases[dataset_key] = phase


with open(r'./results/phases.pkl','rb') as f:
    phases = pickle.load(f)


from my_code.utils.eegpreprocess import eeg_occipital_17_ch
from my_code.utils.megpreprocess import meg_occipital_17_ch

phase_eeg_pick = phases['EEG'][:,:,:,eeg_occipital_17_ch()]
phase_meg_pick = phases['MEG'][:,:,:,meg_occipital_17_ch()]

phase_eeg_pick_mean = np.mean(np.mean(phase_eeg_pick, axis=1),axis=2)
phase_meg_pick_mean = np.mean(np.mean(phase_meg_pick, axis=1),axis=2)


import matplotlib.gridspec as gridspec
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 8
plt.rcParams['font.weight'] = 'bold'
fig = plt.figure(figsize=(4, 4.7),layout='tight')
gs = gridspec.GridSpec(4, 3,width_ratios=[1,1,1], height_ratios=[1,1,1,0.1])
titles = ['(a) 9Hz','(b) 10Hz','(c) 11Hz','(d) 12Hz','(e) 13Hz','(f) 14Hz',
          '(g) 15Hz','(h) 16Hz','(i) 17Hz']
for i,title in zip(range(datasets['MEG'].trial_num),titles):
    # ax = fig.add_subplot(3,3,i+1,projection='polar')
    ax = plt.subplot(gs[i//3,i%3],projection='polar')
    for s in range(datasets['MEG'].sub_num):
        r = [1]*1
        c1 = ax.scatter(phase_eeg_pick_mean[s,i],r,c='#FFC107',marker='v',s=8,alpha=1)
        r = [1.5]*1
        c2 = ax.scatter(phase_meg_pick_mean[s,i],r,c='#FF5733',marker='v',s=8,alpha=1)
        ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2])

        ax.set_xticklabels([])
        ax.set_yticks([1,1.5])
        ax.set_yticklabels([])
        ax.set_ylim([0,2.3])

        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle='-', linewidth=0.8, color='k')
        ax.yaxis.grid(True, linestyle='--', linewidth=0.8, color='k')
        ax.spines['polar'].set_visible(False)

        ax.set_title(title)

        ax.annotate('', xy=(0, 2.3), xytext=(0, 2.2),  
                arrowprops=dict(facecolor='k', edgecolor='black', lw=1,   
                                headlength=3, headwidth=4, width=2))
        ax.annotate('', xy=(np.pi/2, 2.3), xytext=(np.pi/2, 2.2),  
                arrowprops=dict(facecolor='k', edgecolor='black', lw=1,   
                                headlength=3, headwidth=4, width=2))

ax = fig.add_axes([0.1, 0.02, 0.8, 0.1])
ax.axis('off')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.scatter(0.65,0.5,c='#FFC107',marker='v',s=10,alpha=1)
ax.text(0.68,0.44,'EEG',va='center',fontsize=8)

ax.scatter(0.2,0.5,c='#FF5733',marker='v',s=10,alpha=1)
ax.text(0.23,0.44,'OPM-MEG',va='center',fontsize=8)

plt.tight_layout()
plt.show()

# fig.savefig('phase.png',dpi=600)
# fig.savefig('phase.pdf')
# fig.savefig('phase.svg')


# %%
# 相位与刺激频率之间的关系-统计检验
from scipy import stats
stim_freqs = [9,10,11,12,13,14,15,16,17]
stim_phases = [i*np.pi for i in [0, 1, 0, 1.5, 0.5, 1.5, 0, 1, 0]]
phase_error = [np.mean(phase_meg_pick_mean,axis=0)[i] - stim_phases[i] for i in range(9)]
r0, p0 = stats.pearsonr(stim_freqs, phase_error)

phase_error = [np.mean(phase_eeg_pick_mean,axis=0)[i] - stim_phases[i] for i in range(9)]
r1, p1 = stats.pearsonr(stim_freqs, phase_error)

print('MEG:', r0, p0)
print('EEG:', r1, p1)