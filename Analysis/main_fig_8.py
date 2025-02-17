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
from my_code.utils.megpreprocess import myfilterbank as megmyfilterbank
from my_code.utils.eegpreprocess import myfilterbank as eegmyfilterbank
from my_code.utils.megpreprocess import meg_occipital_17_ch
from my_code.utils.eegpreprocess import eeg_occipital_17_ch


eeg_ch_used = eeg_occipital_17_ch()
meg_ch_used = meg_occipital_17_ch()

evaluators = dict()
for key in datasets.keys():
    dataset = datasets[key]
    if key == 'MEG':
        filterbank = megmyfilterbank
        harmonic_num = 7
        ch_used = meg_occipital_17_ch()

    elif key == 'EEG':
        filterbank = eegmyfilterbank
        harmonic_num = 5
        ch_used = eeg_occipital_17_ch()

    dataset.regist_filterbank(filterbank)

    dataset_container = [dataset]
    all_trials = [i for i in range(dataset.trial_num)]

    tw_seq = [i/100 for i in range(20,300+20,20)]

    from SSVEPAnalysisToolbox.evaluator import gen_trials_onedataset_individual_diffsiglen
    trial_container = gen_trials_onedataset_individual_diffsiglen(dataset_idx = 0,
                                                                tw_seq = tw_seq,
                                                                dataset_container = dataset_container,
                                                                harmonic_num = harmonic_num,
                                                                trials = all_trials,
                                                                ch_used = ch_used,
                                                                t_latency = None,
                                                                shuffle = False)

    if key == 'MEG':
        a = 0.75
        b = 0.5
        num_subbands = 5
        suggested_weights = [i**(-a)+b for i in range(1,num_subbands+1,1)]
    elif key == 'EEG':
        a = 0.5
        b = 0
        num_subbands = 5
        suggested_weights = [i**(-a)+b for i in range(1,num_subbands+1,1)]
    
    from SSVEPAnalysisToolbox.algorithms import TRCA,ETRCA,MSETRCA,ITCCA,ECCA,MsetCCA,MSCCA
    model_container = [TRCA(weights_filterbank=suggested_weights),
                       ETRCA(weights_filterbank=suggested_weights),
                       MSETRCA(weights_filterbank=suggested_weights),
                       ECCA(weights_filterbank = suggested_weights),
                       ITCCA(weights_filterbank = suggested_weights),
                       MsetCCA(weights_filterbank = suggested_weights),
                       MSCCA(weights_filterbank = suggested_weights)
                       ]

    from SSVEPAnalysisToolbox.evaluator import BaseEvaluator
    evaluator = BaseEvaluator(dataset_container = dataset_container,
                            model_container = model_container,
                            trial_container = trial_container,
                            save_model = False,
                            disp_processbar = True)

    evaluator.run(n_jobs = 15,eval_train = False)

    evaluators[key] = evaluator


# %%
num_subs = len(dataset.subjects)
for key in evaluators.keys():
    evaluator = evaluators[key]
    dataset = datasets[key]
    from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr
    acc_all = np.zeros((len(model_container),len(tw_seq),num_subs,dataset.block_num))
    itr_all = np.zeros((len(model_container),len(tw_seq),num_subs,dataset.block_num))
    performance_container = evaluator.performance_container
    for i,(trialinfo,performance) in enumerate(zip(trial_container,performance_container)):
        tw = trialinfo[0].tw
        tw_idx = tw_seq.index(tw)

        t_latency = dataset.default_t_latency
        num_targs = len(trialinfo[0].trial_idx[0])

        sub_idx = trialinfo[0].sub_idx[0]
        train_block_idx = trialinfo[0].block_idx[0]

        acc_store = []
        itr_store = []
        for model_idx,model_performance in enumerate(performance):
            Y_test = model_performance.true_label_test
            Y_pred = model_performance.pred_label_test

            acc = cal_acc(Y_true = Y_test, Y_pred = Y_pred)
            itr = cal_itr(tw = tw, t_break = 0, t_latency = t_latency, 
                        t_comp = 0,N = num_targs, acc = acc)

            acc_store.append(acc)
            itr_store.append(itr)

            acc_all[model_idx,tw_idx,sub_idx,i%dataset.block_num] = acc
            itr_all[model_idx,tw_idx,sub_idx,i%dataset.block_num] = itr


    # import pickle
    # if key == 'MEG':
    #     with open(r'.\results\acc_all_meg.pkl','wb') as f:
    #         pickle.dump(acc_all,f)

    #     with open(r'.\results\itr_all_meg.pkl','wb') as f:
    #         pickle.dump(itr_all,f)
    # else:
    #     with open(r'.\results\acc_all_eeg.pkl','wb') as f:
    #         pickle.dump(acc_all,f)

    #     with open(r'.\results\itr_all_eeg.pkl','wb') as f:
    #         pickle.dump(itr_all,f)



# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(2,2,figsize=(6, 5),constrained_layout=True)

import pickle
with open(r'.\results\acc_all_meg.pkl', 'rb') as f:
    acc_all = pickle.load(f)
with open(r'.\results\itr_all_meg.pkl', 'rb') as f:
    itr_all = pickle.load(f)

acc_all = np.mean(acc_all,axis=3)
itr_all = np.mean(itr_all,axis=3)

acc_all_mean = np.mean(acc_all,axis=2)*100
acc_all_std = np.std(acc_all,axis=2)*100
acc_all_stde = acc_all_std / np.sqrt(13)

itr_all_mean = np.mean(itr_all,axis=2)
itr_all_std = np.std(itr_all,axis=2)
itr_all_stde = itr_all_std / np.sqrt(13)

tw_seqs = [i/100 for i in range(20,300+20,20)]

capsize = 3
ms = 3
linewidth = 0.8
colors = ['#FF69B4','#923D3A','#FFA500','#4169E1','#48D1CC','#F08080']
markers = ['o','^','d','s','x','p']
orders = [3,4,5,0,1,2]
labels = ['eCCA','itCCA','ms-eCCA','TRCA','eTRCA','ms-eTRCA']

ax = axes[0,0]
for j in range(6):
    ax.plot(tw_seqs,acc_all_mean[orders[j]],label=labels[j],lw=linewidth,color=colors[j],
            marker=markers[j],markersize=ms)
    ax.errorbar(tw_seqs,acc_all_mean[orders[j]],yerr=acc_all_stde[orders[j]],c=colors[j],
                lw=linewidth,elinewidth=linewidth,capsize=capsize)

ax.hlines(y=90,xmin=0.15,xmax=3.05,color='k',linestyle='--',lw=linewidth)

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel(r'$\regular{T_{train}}$ (s)',fontname='Times New Roman')
ax.set_ylim(0,100)
ax.set_xlim(0.15,3.05)
ax.set_xticks(tw_seqs)
ax.set_xticklabels(['0.2','','0.6','','1.0','','1.4','','1.8','','2.2','','2.6','','3.0'])
ax.set_yticks([0,20,40,60,80,100])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

lg = ax.legend(loc=(0.5,0.05))
lg.set_frame_on(False)

ax = axes[0,1]
for j in range(6):
    ax.plot(tw_seqs,itr_all_mean[orders[j]],label=labels[j],lw=linewidth,color=colors[j],
            marker=markers[j],markersize=ms)
    ax.errorbar(tw_seqs,itr_all_mean[orders[j]],yerr=itr_all_stde[orders[j]],c=colors[j],
                lw=linewidth,elinewidth=linewidth,capsize=capsize)

ax.set_ylabel('ITR (bits/min)')
ax.set_xlabel(r'$\regular{T_{train}}$ (s)',fontname='Times New Roman')
ax.set_ylim(0,160)
ax.set_xlim(0.15,3.05)
ax.set_xticks(tw_seqs)
ax.set_xticklabels(['0.2','','0.6','','1.0','','1.4','','1.8','','2.2','','2.6','','3.0'])
ax.set_yticks([0,40,80,120,160])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# fig.savefig(r'.\fig\fig_8.pdf')
# fig.savefig(r'.\fig\fig_8.png',dpi=600)
# fig.savefig(r'.\fig\fig_8.svg')


import pickle
with open(r'.\results\acc_all_eeg.pkl', 'rb') as f:
    acc_all = pickle.load(f)
with open(r'.\results\itr_all_eeg.pkl', 'rb') as f:
    itr_all = pickle.load(f)

acc_all = np.mean(acc_all,axis=3)
itr_all = np.mean(itr_all,axis=3)

acc_all_mean = np.mean(acc_all,axis=2)*100
acc_all_std = np.std(acc_all,axis=2)*100
acc_all_stde = acc_all_std / np.sqrt(13)

itr_all_mean = np.mean(itr_all,axis=2)
itr_all_std = np.std(itr_all,axis=2)
itr_all_stde = itr_all_std / np.sqrt(13)

tw_seqs = [i/100 for i in range(20,300+20,20)]

capsize = 3
ms = 3
linewidth = 0.8
colors = ['#FF69B4','#923D3A','#FFA500','#4169E1','#48D1CC','#F08080']
markers = ['o','^','d','s','x','p']
orders = [3,4,5,0,1,2]
labels = ['eCCA','itCCA','ms-eCCA','TRCA','eTRCA','ms-eTRCA']

ax = axes[1,0]
for j in range(6):
    ax.plot(tw_seqs,acc_all_mean[orders[j]],label=labels[j],lw=linewidth,color=colors[j],
            marker=markers[j],markersize=ms)
    ax.errorbar(tw_seqs,acc_all_mean[orders[j]],yerr=acc_all_stde[orders[j]],c=colors[j],
                lw=linewidth,elinewidth=linewidth,capsize=capsize)

ax.hlines(y=90,xmin=0.15,xmax=3.05,color='k',linestyle='--',lw=linewidth)

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel(r'$\regular{T_{train}}$ (s)',fontname='Times New Roman')
ax.set_ylim(0,100)
ax.set_xlim(0.15,3.05)
ax.set_xticks(tw_seqs)
ax.set_xticklabels(['0.2','','0.6','','1.0','','1.4','','1.8','','2.2','','2.6','','3.0'])
ax.set_yticks([0,20,40,60,80,100])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

lg = ax.legend(loc=(0.5,0.05))
lg.set_frame_on(False)

ax = axes[1,1]
for j in range(6):
    ax.plot(tw_seqs,itr_all_mean[orders[j]],label=labels[j],lw=linewidth,color=colors[j],
            marker=markers[j],markersize=ms)
    ax.errorbar(tw_seqs,itr_all_mean[orders[j]],yerr=itr_all_stde[orders[j]],c=colors[j],
                lw=linewidth,elinewidth=linewidth,capsize=capsize)

ax.set_ylabel('ITR (bits/min)')
ax.set_xlabel(r'$\regular{T_{train}}$ (s)',fontname='Times New Roman')
ax.set_ylim(0,160)
ax.set_xlim(0.15,3.05)
ax.set_xticks(tw_seqs)
ax.set_xticklabels(['0.2','','0.6','','1.0','','1.4','','1.8','','2.2','','2.6','','3.0'])
ax.set_yticks([0,40,80,120,160])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


fig.text(0,0.96,'A',fontweight='bold',fontsize=16)
fig.text(0.5,0.96,'B',fontweight='bold',fontsize=16)
fig.text(0,0.46,'C',fontweight='bold',fontsize=16)
fig.text(0.5,0.46,'D',fontweight='bold',fontsize=16)

plt.show()

# fig.savefig(r'.\fig\fig_8.pdf')
# fig.savefig(r'.\fig\fig_8.png',dpi=600)
# fig.savefig(r'.\fig\fig_8.svg')
