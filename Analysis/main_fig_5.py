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
from my_code.datasets.mybenchmarkdataset import MyBenchmarkDataset
from my_code.datasets.mybetadataset import MyBetaDataset

import my_code.utils.megpreprocess as megpreproc
import my_code.utils.eegpreprocess as eegpreproc
import SSVEPAnalysisToolbox.utils.benchmarkpreprocess as benchmarkpreproc
import my_code.utils.betapreprocess as betapreproc

from my_code.utils.megpreprocess import bandpass_filter
from my_code.utils.meegutils import find_nearest_index, amplitude_spectrum, snr_spectrum

projct_path = os.getcwd()

MEG = MyMEGDataset(path='datasets/OPMEEGBCI/MEG')
EEG = MyEEGDataset(path='datasets/OPMEEGBCI/EEG')
Benchmark = MyBenchmarkDataset(path=os.path.join(projct_path,'datasets','Benchmark'),
                             path_support_file=os.path.join(projct_path,'datasets','Benchmark'))
Beta = MyBetaDataset(path=os.path.join(projct_path,'datasets','Beta'))

MEG.regist_preprocess(megpreproc.preprocess)
EEG.regist_preprocess(eegpreproc.preprocess)
Benchmark.regist_preprocess(benchmarkpreproc.preprocess)
Beta.regist_preprocess(betapreproc.preprocess)

datasets = {'MEG':MEG,'EEG':EEG,'Benchmark':Benchmark,'Beta':Beta}

# %%
sfreq = MEG.srate
stim_id = MEG.stim_info['stim_id']
stim_freqs = MEG.stim_info['freqs']
stim_phases = MEG.stim_info['phases']
stim_dict = dict()
for i in range(len(stim_id)):
    stim_dict[str(stim_freqs[i])+'Hz'] = stim_id[i]


# %%----------------------------------------fig 5A----------------------------------------

spectrum = dict()
for dataset_key in ['MEG','EEG']:
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

# 绘制频谱的头部地形图
info_meg = datasets['MEG'].get_sub_info(sub_idx=0)
info_eeg = datasets['EEG'].get_sub_info(sub_idx=0)
from matplotlib.colors import LinearSegmentedColormap
vlims = [[(-10,60),(-10,60),(-10,60),(-10,60),(-10,60),(-10,60),(-10,60),(-10,60),(-10,60)],
         [(0,0.9),(0,0.8),(0,0.8),(0,0.7),(0,0.7),(0,0.6),(0,0.5),(0,0.5),(0,0.5)]]
spheres = [0.125,0.1]
for key,info,vlim,sphere in zip(spectrum.keys(),[info_meg,info_eeg],vlims, spheres):
    plt.rcParams['font.family'] = 'Times New Roman,SimSun'
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'bold'
    fig = plt.figure(figsize=(3, 3))
    gs = gridspec.GridSpec(3, 3,width_ratios=[1,1,1])

    for i in range(len(stim_freqs)):
        stim_freq = stim_freqs[i]
        freq_idx = find_nearest_index(spectrum[key]['f'], stim_freq)

        ax = plt.subplot(gs[i//3,i%3])
        ax.set_aspect('equal', adjustable='box')
        im,_ = mne.viz.plot_topomap(spectrum[key]['spectrum'][i,:,freq_idx],info,axes=ax,
                                    contours=5,sphere=sphere,
                                    vlim=vlim[i],cmap='RdBu_r',extrapolate='head')

    plt.tight_layout()
    plt.show()

# %%----------------------------------------fig 5B----------------------------------------

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

picks_meg = ['PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']
picks_eeg = ['PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']
picks_benchmark = ['PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']
picks_bata = ['PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']

picks_freqs = [9,10,11,12,13,14,15]

picks_meg_idx = [datasets['MEG'].channels.index(pick_ch) for pick_ch in picks_meg]
picks_eeg_idx = [datasets['EEG'].channels.index(pick_ch) for pick_ch in picks_eeg]
picks_benchmark_idx = [datasets['Benchmark'].channels.index(pick_ch) for pick_ch in picks_benchmark]
picks_bata_idx = [datasets['Beta'].channels.index(pick_ch) for pick_ch in picks_bata]

picks_meg_freqs_idx = [datasets['MEG'].stim_info['freqs'].index(pick_freq) for pick_freq in picks_freqs]
picks_eeg_freqs_idx = [datasets['EEG'].stim_info['freqs'].index(pick_freq) for pick_freq in picks_freqs]
picks_benchmark_freqs_idx = [datasets['Benchmark'].stim_info['freqs'].index(pick_freq) for pick_freq in picks_freqs]
picks_beta_freqs_idx = [datasets['Beta'].stim_info['freqs'].index(pick_freq) for pick_freq in picks_freqs]

snr_fft_meg_pick = snrs_fft['MEG'][:,:,picks_meg_freqs_idx,:][:,:,:,picks_meg_idx]
snr_fft_eeg_pick = snrs_fft['EEG'][:,:,picks_eeg_freqs_idx,:][:,:,:,picks_eeg_idx]
snr_fft_benchmark_pick = snrs_fft['Benchmark'][:,:,picks_benchmark_freqs_idx,:][:,:,:,picks_benchmark_idx]
snr_fft_beta_pick = snrs_fft['Beta'][:,:,picks_beta_freqs_idx,:][15:,:,:,picks_bata_idx]


eeg_color = '#EC8892'
meg_color = '#FAB85A'
benchmark_color = '#28A49A'
beta_color = '#5048BC'

colors = [meg_color,eeg_color,benchmark_color,beta_color]

import matplotlib.gridspec as gridspec
from scipy.stats import norm

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'normal'
fig = plt.figure(figsize=(4.5, 3),layout='tight')
gs = gridspec.GridSpec(1, 1,width_ratios=[1])
ax = plt.subplot(gs[0,0])

# 计算直方图数据
bins = np.arange(-30,-15,0.25)
alpha = 0.6

bars = []
for i,data in enumerate([snr_fft_meg_pick,snr_fft_eeg_pick,snr_fft_benchmark_pick,snr_fft_beta_pick]):
    data = data.flatten()
    color = colors[i]
    counts, bins = np.histogram(data, bins=bins,normed=True)
    bar_width = 0.8 * (bins[1] - bins[0])
    b= ax.bar(bins[:-1], counts,color=color,width=bar_width,
        align='edge',edgecolor=None,alpha=alpha)

    bars.append(b)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(data), np.std(data))
    plt.plot(x, p, color, linewidth=1.5)
    plt.axvline(np.mean(data), color=color, linestyle='--')


ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#C0C0C0')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#C0C0C0')

ax.set_xlim([bins[0],bins[-1]])
ax.set_xticks(np.arange(bins[0],bins[-1]+3,3))

ax.set_ylim(0,0.27)

ax.set_xlabel('SNR(dB)',fontweight='normal')
ax.set_ylabel('Probability',fontweight='normal')

lg = ax.legend(bars,['OPM-MEG','EEG','Beta','Benchmark'])
lg.set_frame_on(False)

plt.show()


# fig.savefig(r'.\fig\snr_histogram.png',dpi=600)
# fig.savefig(r'.\fig\snr_histogram.pdf')
# fig.savefig(r'.\fig\snr_histogram.svg')


# %%----------------------------------------fig 5C----------------------------------------
spec_snr = dict()
for dataset_key in datasets.keys():
    dataset = datasets[dataset_key]
    if dataset.ID == 'MEG':
        data_pick = dataset.get_data_single_channel(chs=['O2','O1','POZ','OZ'])
    elif dataset.ID == 'EEG':
        data_pick = dataset.get_data_single_channel(chs=['O2','O1','POZ','OZ'])

    data_pick_Hz = []
    snrs = []
    specs = []
    ratios = []
    for d_pick in data_pick:
        data_pick_Hz.append(np.mean(d_pick[:,4,:,:],axis=0))
        data_pick_mean = np.mean(d_pick,axis=0)
        snrs_sub = []
        specs_sub = []
        ratio_sub = []
        for ch in range(data_pick_mean.shape[1]):
            data_pick_mean_ch = data_pick_mean[:,ch,:]
            lowcut = 7.0  # 低截止频率
            highcut = 90.0  # 高截止频率
            data_pick_mean_ch_filt = bandpass_filter(dataset, data_pick_mean_ch, lowcut, highcut)

            # Oz通道的幅度谱
            f,amp_spectrum_pick_ch = amplitude_spectrum(dataset,data_pick_mean_ch_filt)

            snr = snr_spectrum(amp_spectrum_pick_ch, noise_n_neighbor_freqs=20, noise_skip_neighbor_freqs=5)

            specs_sub.append(amp_spectrum_pick_ch)
            snrs_sub.append(snr)

        specs_sub_mean = np.mean(specs_sub,axis=0)
        snrs_sub_mean = np.mean(snrs_sub,axis=0)

        specs.append(specs_sub_mean)
        snrs.append(snrs_sub_mean)

    data_pick_Hz_mean = np.mean(np.mean(data_pick_Hz,axis=0),axis=0)
    snrs_mean = np.mean(snrs,axis=0)
    specs_mean = np.mean(specs,axis=0)

    spec_snr[dataset.ID] = dict()
    spec_snr[dataset.ID]['f'] = f
    spec_snr[dataset.ID]['specs_mean'] = specs_mean
    spec_snr[dataset.ID]['snrs_mean'] = snrs_mean

import matplotlib.gridspec as gridspec
import scicomap as sc
from matplotlib.colors import LinearSegmentedColormap
cmap = sc.ScicoSequential(cmap='afmhot')
cmap = cmap.get_mpl_color_map()


colors = ['White', '#CD5C5C']
cmap_meg = LinearSegmentedColormap.from_list('custom_cmap', colors)
colors = ['White', '#4682B4']
cmap_eeg = LinearSegmentedColormap.from_list('custom_cmap', colors)

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'normal'
fig = plt.figure(figsize=(11, 4.5))
gs = gridspec.GridSpec(2, 2,width_ratios=[1,1])
axes = [plt.subplot(gs[0,0]),
        plt.subplot(gs[1,0]),
        plt.subplot(gs[0,1]),
        plt.subplot(gs[1,1])]

vlims = [[0,30],[0,0.6]]
cmaps = [cmap_meg,cmap_eeg]
titles = ['(a) OPM-MEG spectrum',
          '(c) OPM-MEG SNR',
          '(b) EEG spectrum',
          '(d) EEG SNR']
for i,(key,vlim,cmap) in enumerate(zip(datasets.keys(),vlims,cmaps)):
    dataset = datasets[key]
    f = spec_snr[key]['f']
    specs_mean = spec_snr[key]['specs_mean']
    snrs_mean = spec_snr[key]['snrs_mean']

    ax = axes[i*2]
    im = ax.imshow(specs_mean,aspect='auto',cmap=cmap,origin='lower',
                   vmin=vlim[0],vmax=vlim[1])
    ax.set_title(titles[i*2],fontsize=11)
    ax.set_ylabel(' ')
    ax.set_xlabel('Response Frequency (Hz)',fontweight='normal',fontsize=11)
    ax.set_yticks(np.arange(dataset.trial_num))
    ax.set_yticklabels(stim_freqs)
    f_pick = [7,90]
    f_pick_idx = [find_nearest_index(f, i) for i in f_pick]
    ax.set_xlim(f_pick_idx)
    f_pick = [7,20,30,40,50,60,70,80,90]
    f_pick_idx = [find_nearest_index(f, i) for i in f_pick]
    ax.set_xticks(f_pick_idx)
    ax.set_xticklabels(f_pick)
    cbar = plt.colorbar(im, ax=ax, location='right', fraction=0.046, pad=0.04)
    cbar.set_ticks(np.linspace(vlim[0],vlim[1],5))
    cbar.set_label(f'Amplitude spectra ({dataset.unit})', fontsize=11, fontweight='normal')


    ax = axes[i*2+1]
    im = ax.imshow(snrs_mean,aspect='auto',cmap=cmap,origin='lower',vmin=0,vmax=4)
    ax.set_title(titles[i*2+1],fontsize=11)
    ax.set_xlabel('Response Frequency (Hz)',fontweight='normal',fontsize=11)
    ax.set_ylabel(' ')
    ax.set_yticks(np.arange(dataset.trial_num))
    ax.set_yticklabels(stim_freqs)
    f_pick = [7,90]
    f_pick_idx = [find_nearest_index(f, i) for i in f_pick]
    ax.set_xlim(f_pick_idx)
    f_pick = [7,20,30,40,50,60,70,80,90]
    f_pick_idx = [find_nearest_index(f, i) for i in f_pick]
    ax.set_xticks(f_pick_idx)
    ax.set_xticklabels(f_pick)
    cbar = plt.colorbar(im, ax=ax, location='right', fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1,2,3,4])
    cbar.set_label('SNR (dB)', fontsize=12, fontweight='normal')


fig.text(0.02, 0.5, 'Stimulation Frequency (Hz)', ha='center', va='center', 
         rotation='vertical',fontweight='normal',fontsize=11)
fig.text(0.52, 0.5, 'Stimulation Frequency (Hz)', ha='center', va='center', 
         rotation='vertical',fontweight='normal',fontsize=11)


plt.tight_layout()
plt.show()


# %%
