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
import numpy as np
import scipy.io as scio
import os.path as op
import os


import mne
from mne.io import curry
from mne import transforms
from mne.simulation import simulate_raw,simulate_sparse_stc
from mne.viz.backends.renderer import _get_renderer
from mne.transforms import apply_trans,_get_trans,_get_transforms_to_coord_frame,_frame_to_str
from mne.io.pick import _picks_to_idx

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne._fiff.pick import channel_indices_by_type,_DATA_CH_TYPES_SPLIT
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from mne.viz.backends.renderer import _get_renderer
from mne.transforms import apply_trans
from mne._fiff.pick import channel_indices_by_type,_DATA_CH_TYPES_SPLIT
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from mne.viz.topomap import _get_pos_outlines
from mne.viz.utils import _check_sphere
def _draw_outlines(ax, outlines,color='k',linewidth=1):
    """Draw the outlines for a topomap."""
    from matplotlib import rcParams

    outlines_ = {k: v for k, v in outlines.items() if k not in ["patch"]}
    for key, (x_coord, y_coord) in outlines_.items():
        if "mask" in key or key in ("clip_radius", "clip_origin"):
            continue
        ax.plot(
            x_coord,
            y_coord,
            color=color,
            # color=rcParams["axes.edgecolor"],
            linewidth=linewidth,
            clip_on=False,
        )
    return outlines_

def plot_sensors2d(info,sphere=None,show_names=False,axes=None,figsize=(4,4),outlinewidth=1,
                     mask=None,mask_params=None):
    
    ch_indices = channel_indices_by_type(info)
    picks = list()
    allowed_types = _DATA_CH_TYPES_SPLIT
    for this_type in allowed_types:
        picks += ch_indices[this_type]

    dev_head_t = info["dev_head_t"]
    chs = info["chs"]
    pos = np.empty((len(chs), 3))
    for ci, ch in enumerate(chs):
        pos[ci] = ch["loc"][:3]
        if ch["coord_frame"] == mne.io.constants.FIFF.FIFFV_COORD_DEVICE:
            pos[ci] = apply_trans(dev_head_t, pos[ci])

    ch_names = info.ch_names
    num_chans = len(picks)
    sphere = _check_sphere(sphere, info)
    pos, outlines = _get_pos_outlines(info, picks, sphere, to_sphere=True)
    if axes is None:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['font.weight'] = 'bold'
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    else:
        ax = axes
        fig = ax.get_figure()
    _draw_outlines(ax, outlines,linewidth=outlinewidth)


    poss = []
    show_names = []
    for ms in mask:
        picks = _picks_to_idx(info,picks=ms,exclude=[])
        mask1 = np.array([True if i in picks else False for i in range(num_chans) ])
        pos1 = pos[mask1]
        show_name1 = [ch_names[i] for i,b in enumerate(mask1) if b]

        poss.append(pos1)
        show_names.append(show_name1)


    for i in range(len(mask)):
        mask_param = mask_params[i]
        show_name = show_names[i]
        pos = poss[i]

        marker = mask_param.get("marker", 'o')
        marker_s = mask_param.get("markersize")
        marker_facecolors = mask_param.get("markerfacecolor", None)
        marker_edgecolors = mask_param.get("markeredgecolor", None)
        marker_linewidths = mask_param.get("linewidth", 1)
        marker_alpha = mask_param.get("markeralpha", 1)

        text_alpha = mask_param.get("textalpha", 1)
        text_color = mask_param.get("textcolor", 'k')
        text_fontsize = mask_param.get("fontsize", None)
        text_fontstyle =  mask_param.get("fontstyle", None)
        text_fontweight = mask_param.get("fontweight", None)


        ax.scatter(pos[:, 0],pos[:, 1],picker=True,clip_on=False,marker=marker,
                        c=None,s=marker_s,linewidths=marker_linewidths,alpha=marker_alpha,
                        facecolors=marker_facecolors, edgecolors=marker_edgecolors,
                        plotnonfinite=False)
        
        indices = range(len(pos))
        for idx in indices:
            this_pos = pos[idx]
            ax.text(this_pos[0],this_pos[1],show_name[idx],
                ha="center",va="center",
                alpha=text_alpha,color=text_color,fontsize=text_fontsize,
                fontweight=text_fontweight,fontstyle=text_fontstyle
            )


    ax.set(aspect="equal")
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig


info_meg = datasets['MEG'].get_sub_info(sub_idx=0)
info_eeg = datasets['EEG'].get_sub_info(sub_idx=0)

eeg_mask = [['FP1','FP2','AF3','AF4','FPZ','FZ','F1','F2','F3','F4','F5','F6','F7','F8'],
            ['FCZ','FC1','FC2','FC3','FC4','CZ','C1','C2','C3','C4','CPZ','CP1','CP2','CP3','CP4'],
            ['T7','T8','FC5','FC6','C5','C6','CP5','CP6','TP7','TP8'],
            ['P7','P8','P5','P6','P3','P4','P1','P2','PZ','POZ','PO3','PO4','PO7','PO8','O1','O2','OZ']]

meg_mask = [['FP1','FP2','AF3','AF4','AF7','AF8','AFZ','FPZ','FZ','F1','F2','F3','F4','F5','F6','F7','F8'],
            ['FCZ','FC1','FC2','FC3','FC4','CZ','C1','C2','C3','C4','CPZ','CP1','CP2','CP3','CP4'],
            ['FT7','FT8','T7','T8','FC5','FC6','C5','C6','CP5','CP6','TP7','TP8'],
            ['P7','P8','P5','P6','P3','P4','P1','P2','PZ','POZ','PO3','PO4','PO7','PO8','P9','P10',
             'O1','O2','OZ','IZ']]

alpha = 1
linewidth = 1
markersize = 280
fontsize = 9
color = ['#fa86a9','#b0d097','#fcc351','#71bcec']
meg_mask_param = [{'markersize':markersize,'markeralpha':alpha,'marker':'s',
               'markeredgecolor':color[0],'markerfacecolor':color[0],'linewidth':linewidth,
               'fontsize':fontsize},
               {'markersize':markersize,'markeralpha':alpha,'marker':'s',
               'markeredgecolor':color[1],'markerfacecolor':color[1],'linewidth':linewidth,
               'fontsize':fontsize},
               {'markersize':markersize,'markeralpha':alpha,'marker':'s',
               'markeredgecolor':color[2],'markerfacecolor':color[2],'linewidth':linewidth,
               'fontsize':fontsize},
               {'markersize':markersize,'markeralpha':alpha,'marker':'s',
               'markeredgecolor':color[3],'markerfacecolor':color[3],'linewidth':linewidth,
               'fontsize':fontsize}]

eeg_mask_param = [{'markersize':markersize,'markeralpha':alpha,'marker':'o',
               'markeredgecolor':color[0],'markerfacecolor':color[0],'linewidth':linewidth,
               'fontsize':fontsize},
               {'markersize':markersize,'markeralpha':alpha,'marker':'o',
               'markeredgecolor':color[1],'markerfacecolor':color[1],'linewidth':linewidth,
               'fontsize':fontsize},
               {'markersize':markersize,'markeralpha':alpha,'marker':'o',
               'markeredgecolor':color[2],'markerfacecolor':color[2],'linewidth':linewidth,
               'fontsize':fontsize},
               {'markersize':markersize,'markeralpha':alpha,'marker':'o',
               'markeredgecolor':color[3],'markerfacecolor':color[3],'linewidth':linewidth,
               'fontsize':fontsize}]

plt.rcParams['font.family'] = 'Times New Roman,SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
fig,axes = plt.subplots(1,2,figsize=(9, 5),constrained_layout=True)
fig.patch.set_facecolor((1, 1, 1, 0))


plot_sensors2d(info_meg,sphere=0.1,show_names=True,axes=axes[0],mask=meg_mask,mask_params=meg_mask_param)
axes[0].set_facecolor((1, 1, 1, 0))

plot_sensors2d(info_eeg,sphere=0.1,show_names=True,axes=axes[1],mask=eeg_mask,mask_params=eeg_mask_param)
plt.show()

# fig.savefig('sensor.svg')

# %%
