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
import my_code.megpreprocess as megpreproc
import my_code.eegpreprocess as eegpreproc
from my_code.datasets.mymegdataset import MyMEGDataset
from my_code.datasets.myeegdataset import MyEEGDataset

data_path = r"D:\科研\代码\工作\6、BCI_Data\datasets\OPMEEGBCI\MEG"
MEG = MyMEGDataset(path=data_path)

data_path = r"D:\科研\代码\工作\6、BCI_Data\datasets\OPMEEGBCI\EEG"
EEG = MyEEGDataset(path=data_path)

MEG.regist_preprocess(megpreproc.preprocess)
EEG.regist_preprocess(eegpreproc.preprocess)

datasets = {'MEG':MEG,'EEG':EEG}

# %%
from my_code.eegpreprocess import eeg_occipital_17_ch
from my_code.megpreprocess import meg_occipital_17_ch
from my_code.myfilterbank import myfilterbank4

evaluators = dict()
for idx,key in enumerate(datasets.keys()):
    dataset = datasets[key]
    num_targs = dataset.stim_info['stim_num']
    stim_freqs = dataset.stim_info['freqs']
    all_stims = [i for i in range(dataset.trial_num)]
    num_subs = len(dataset.subjects)
    num_trials = dataset.block_num
    labels = np.arange(num_targs)

    if key == 'MEG':
        a = 0.75
        b = 0.5
        num_subbands = 4
        weights_filterbank = [i**(-a)+b for i in range(1,num_subbands+1,1)]
        harmonic_num = 4
        ch_used = meg_occipital_17_ch()
        dataset.regist_filterbank(myfilterbank4)
    elif key == 'EEG':
        a = 0.5
        b = 0
        num_subbands = 4
        weights_filterbank = [i**(-a)+b for i in range(1,num_subbands+1,1)]
        harmonic_num = 4
        ch_used = eeg_occipital_17_ch()
        dataset.regist_filterbank(myfilterbank4)

    num_trains = [1]
    tw_seq = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]

    # List shape: num_tw*num_subs*num_trains*num_cv
    trial_container = dataset.gen_trials_leave_out(tw_seq = tw_seq,
                                                trains = num_trains,
                                                harmonic_num = harmonic_num,
                                                ch_used = ch_used)

    from SSVEPAnalysisToolbox.algorithms import ETRCA
    from my_code.algorithms.ress import ERESS
    from my_code.algorithms.prca import EPRCA

    model_container = [
                        EPRCA(weights_filterbank=weights_filterbank,
                            stim_freqs=stim_freqs,srate=dataset.srate),
                        ERESS(weights_filterbank=weights_filterbank,
                            stim_freqs=stim_freqs,
                            srate=dataset.srate,
                            ress_param={'peakwidt':0.75, 'neighfreq':3, 'neighwidt':3}),
                        ETRCA(weights_filterbank=weights_filterbank),
                    ]

    from SSVEPAnalysisToolbox.evaluator import BaseEvaluator
    evaluator = BaseEvaluator(dataset_container = [dataset],
                            model_container = model_container,
                            trial_container = trial_container,
                            save_model = False,
                            disp_processbar = True)

    evaluator.run(n_jobs = 20,eval_train = False)

    evaluators[key] = evaluator

# %%
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

        print(i)
        print(f'tw:{tw}    sub:{sub_idx}    num_train:{len(train_block_idx)}')
        print(f'EPRCA  acc:{acc_store[0]*100:.2f}%    itr: {itr_store[0]:.2f} bits/min')
        print(f'ERESS  acc:{acc_store[1]*100:.2f}%    itr: {itr_store[1]:.2f} bits/min')
        print(f'ETRCA  acc:{acc_store[2]*100:.2f}%    itr: {itr_store[2]:.2f} bits/min')
        # print(f'EPRESS  acc:{acc_store[3]*100:.2f}%    itr: {itr_store[3]:.2f} bits/min\n')

    # import pickle
    # if key == 'MEG':
    #     with open(r'.\result\acc_all_meg.pkl','wb') as f:
    #         pickle.dump(acc_all,f)

    #     with open(r'.\result\itr_all_meg.pkl','wb') as f:
    #         pickle.dump(itr_all,f)
    # else:
    #     with open(r'.\result\acc_all_eeg.pkl','wb') as f:
    #         pickle.dump(acc_all,f)

    #     with open(r'.\result\itr_all_eeg.pkl','wb') as f:
    #         pickle.dump(itr_all,f)



# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'
fig,axes = plt.subplots(1,4,figsize=(10, 2.5),constrained_layout=True)
for i,key in enumerate(['MEG','EEG']):
    import pickle
    if key == 'MEG':
        with open(r'.\results\acc_all_meg_single_trial.pkl', 'rb') as f:
            acc_all = pickle.load(f)
        with open(r'.\results\itr_all_meg_single_trial.pkl', 'rb') as f:
            itr_all = pickle.load(f)
    else:
        with open(r'.\results\acc_all_eeg_single_trial.pkl', 'rb') as f:
            acc_all = pickle.load(f)
        with open(r'.\results\itr_all_eeg_single_trial.pkl', 'rb') as f:
            itr_all = pickle.load(f)

    acc_all = np.mean(acc_all,axis=3)
    itr_all = np.mean(itr_all,axis=3)

    acc_all_mean = np.mean(acc_all,axis=2)*100
    acc_all_std = np.std(acc_all,axis=2)*100
    acc_all_stde = acc_all_std / np.sqrt(13)

    itr_all_mean = np.mean(itr_all,axis=2)
    itr_all_std = np.std(itr_all,axis=2)
    itr_all_stde = itr_all_std / np.sqrt(13)

    tw_seqs = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]

    capsize = 3
    ms = 3
    linewidth = 0.8
    colors = ['#FF69B4','#4169E1','#48D1CC','#F08080']

    ax = axes[i*2]
    ax.plot(tw_seqs,acc_all_mean[0],label='ePRCA',lw=linewidth,color=colors[0],
            marker='o',markersize=ms)
    ax.errorbar(tw_seqs,acc_all_mean[0],yerr=acc_all_stde[0],c=colors[0],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)
    print(acc_all_mean[0])

    ax.plot(tw_seqs,acc_all_mean[1],label='eRESS',lw=linewidth,color=colors[1],
            marker='^',markersize=ms)
    ax.errorbar(tw_seqs,acc_all_mean[1],yerr=acc_all_stde[1],c=colors[1],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)

    ax.plot(tw_seqs,acc_all_mean[2],label='eTRCA',lw=linewidth,color=colors[2],
            marker='d',markersize=ms)
    ax.errorbar(tw_seqs,acc_all_mean[2],yerr=acc_all_stde[2],c=colors[2],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)



    ax.hlines(y=90,xmin=0.15,xmax=2.05,color='k',linestyle='--',lw=linewidth)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel(r'$\regular{T_{train}}$ (s)',fontname='Times New Roman')
    ax.set_ylim(0,100)
    ax.set_xlim(0.15,2.05)
    ax.set_xticks(tw_seqs)
    ax.set_yticks([0,20,40,60,80,100])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



    ax = axes[i*2+1]
    ax.plot(tw_seqs,itr_all_mean[0],label='ePRCA',lw=linewidth,color=colors[0],
            marker='o',markersize=ms)
    ax.errorbar(tw_seqs,itr_all_mean[0],yerr=itr_all_stde[0],c=colors[0],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)
    
    print(itr_all_mean[0])

    ax.plot(tw_seqs,itr_all_mean[1],label='eRESS',lw=linewidth,color=colors[1],
            marker='^',markersize=ms)
    ax.errorbar(tw_seqs,itr_all_mean[1],yerr=itr_all_stde[1],c=colors[1],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)

    ax.plot(tw_seqs,itr_all_mean[2],label='eTRCA',lw=linewidth,color=colors[2],
            marker='d',markersize=ms)
    ax.errorbar(tw_seqs,itr_all_mean[2],yerr=itr_all_stde[2],c=colors[2],lw=linewidth,
                elinewidth=linewidth,capsize=capsize)


    ax.set_ylabel('ITR (bits/min)')
    ax.set_xlabel(r'$\regular{T_{train}}$ (s)',fontname='Times New Roman')
    ax.set_ylim(0,140)
    ax.set_xlim(0.15,2.05)
    ax.set_xticks(tw_seqs)
    ax.set_yticks([0,20,40,60,80,100,120,140])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    lg = ax.legend(loc=(0.65,0.7),fontsize=8)
    lg.set_frame_on(False)

fig.text(0,0.93,'A',fontweight='bold',fontsize=18)
fig.text(0.5,0.93,'B',fontweight='bold',fontsize=18)

plt.show()

# fig.savefig('fig_10.png',dpi=900)
# fig.savefig('fig_10.pdf')
# fig.savefig('fig_10.svg')


