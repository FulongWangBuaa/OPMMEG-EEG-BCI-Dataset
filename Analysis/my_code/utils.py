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

import sys
sys.path.append("D:\科研\代码\函数")
from wfl_plot_alignment import wfl_plot_alignment

def read_eeg_data(data_path,trigger_path,sensor_path):
    """
    Reads EEG data from a file and returns it as a mne.io.Raw or mne.io.RawArray.
    """
    curry.curry.FILE_EXTENSIONS['Curry 8'].update({'info': '.cdt.dpo','labels': '.cdt.dpo'})
    raw = mne.io.read_raw_curry(data_path, preload=True)
    raw.info["line_freq"] = 50.0

    eeg_trigger = np.loadtxt(trigger_path,dtype='int')

    raw_pick = raw.copy().pick(picks=['eeg','stim'], 
                          exclude=['F11','F12','FT12','FT11','M1','M2','Cb1','Cb2'])
    montage = mne.channels.read_custom_montage(sensor_path)

    ch_names_raw = raw_pick.ch_names
    ch_names_montage = list(montage.get_positions()['ch_pos'].keys())

    ch_names_raw_upper = [name.upper() for name in raw_pick.ch_names]
    ch_names_montage_upper = [name.upper() for name in list(montage.get_positions()['ch_pos'].keys())]

    chs_num = len(raw_pick.ch_names)
    mapping = dict()
    for name in ch_names_raw_upper:
        idx_in_raw = ch_names_raw_upper.index(name)
        idx_in_montage = ch_names_montage_upper.index(name)

        montage.rename_channels({ch_names_montage[idx_in_montage]: name})

        mapping[ch_names_raw[idx_in_raw]] = name

    raw_pick.rename_channels(mapping)
    raw_pick.set_montage(montage)

    raw = raw_pick.copy()

    # trigger_9 = list(eeg_trigger[0:9,1])
    true_id = [173,178,183,188,193,198,203,208,213]
    fault_id = [218,166,246,158,193,177,233,133,213]

    annotations = raw.annotations
    sfreq = raw.info['sfreq']
    events = []
    for anno in annotations:
        if anno['duration'] == 0 and (int(anno['description']) in fault_id or int(anno['description']) in true_id):
            event = np.zeros((3), dtype=int)
            event[0] = int(anno['onset'] * sfreq)
            event[1] = 1
            event[2] = int(anno['description'])
            events.append(event)

    events = np.array(events)

    from collections import Counter 
    count = Counter([i for i in events[:,2]])
    result = [item for item, cnt in count.items() if cnt > 10]
    
    if set(true_id) == set(result):
        fault_id = [173,178,183,188,193,198,203,208,213]

    if events.shape[0] > 135:
        events_filtered = []
        j = 0
        for i,e in enumerate(events):
            id_in_anno = true_id[fault_id.index(e[2])]
            id_in_txt = eeg_trigger[j,1]
            if id_in_anno == id_in_txt:
                events_filtered.append(i)  # 保留符合条件的事件
                j = j + 1
            else:
                continue
    else:
        events_filtered = list(range(events.shape[0]))

    onset = []
    duration = []
    description = []
    idx = 0
    for anno_idx,anno in enumerate(annotations):
        if anno['duration'] == 0 and int(anno['description']) in fault_id:
            if idx in events_filtered:
                onset.append(anno['onset'])
                duration.append(anno['duration'])
                description.append(str(true_id[fault_id.index(int(anno['description']))]))
            idx += 1
    
    annotations_new = mne.Annotations(onset, duration, description)
    raw.set_annotations( mne.Annotations([],[],[]))
    raw.set_annotations(annotations_new)


    events_id = [173,178,183,188,193,198,203,208,213]
    annotations = raw.annotations
    sfreq = raw.info['sfreq']
    events = []
    i = 0
    for anno in annotations:
        if anno['duration'] == 0 and int(anno['description']) in events_id:
            i += 1
            event = np.zeros((3), dtype=int)
            event[0] = int(anno['onset'] * sfreq)
            event[1] = 1
            event[2] = int(anno['description'])
            events.append(event)
    events = np.array(events)

    return raw, events

def read_meg_data(data_path,trigger_path,sensor_path):
    fs = 1000  # 采样率1000Hz
    n_record_chans = 66 # 记录通道数
    file_id = open(data_path, "rb")  # 二进制文件
    baseDate_data = np.fromfile(file_id, dtype=np.float32)  # 读取二进制文件
    General_Time_In_Seconds = len(baseDate_data) // n_record_chans // fs  # 读取数据的轮数
    Single_Sensor_Data_Length = General_Time_In_Seconds * fs  # 单探头的数据量
    file_id.close()  # 关闭文件
    read_raw_data = np.zeros((n_record_chans, Single_Sensor_Data_Length))  # 创建一个空数组，用于存放拼接后的数据
    for channel_index in range(n_record_chans):  # 遍历探头（通道数）
        for time_seconds in range(General_Time_In_Seconds):  # 遍历记录轮数(每秒的数据）
            # 将baseDate_data中对应位置的数据赋值给All_Channel_Data中对应位置
            read_raw_data[channel_index, time_seconds * fs:(time_seconds + 1) * fs] = baseDate_data[channel_index * fs + (time_seconds * n_record_chans * fs):(channel_index + 1) * fs + (time_seconds * n_record_chans * fs)]

    use_chans = 65
    raw_data = read_raw_data[:use_chans, :]
    raw_data[:-1, :] = raw_data[:-1, :] * 1e-12

    num_chans_data = 64
    sensor_info = scio.loadmat(sensor_path)

    label = list(sensor_info['ch_names'])
    label = [lab.strip() for lab in label]
    pos = sensor_info['pos']
    ori = sensor_info['ori']

    sfreq = 1000
    num_chan = 64
    raw_info = mne.create_info(
    ch_names = label + ['Trigger'],
    ch_types = ['eeg' for i in range(num_chan)] + ['stim'],sfreq=sfreq)

    data_project = os.path.dirname(data_path)
    files_list = os.listdir(data_project)
    if 'bad_segments.txt' in files_list:
        print('bad segments found')
        bad_segments = np.loadtxt(op.join(data_project,'bad_segments.txt'),dtype='int')
        raw_data = np.concatenate(( raw_data[:, :bad_segments[0]*1000],raw_data[:,bad_segments[1]*1000:]), axis=1)

    raw = mne.io.RawArray(raw_data, raw_info)

    dic = {label[i]: pos[i,:] for i in range(num_chans_data)}
    montage = mne.channels.make_dig_montage(ch_pos=dic, coord_frame='head')
    raw = raw.set_montage(montage)

    for j,ch_name in enumerate(raw.info['ch_names']):
        if ch_name != 'Trigger':
            raw.info['chs'][j]['kind'] = mne.io.constants.FIFF.FIFFV_MEG_CH # 通道类型
            raw.info['chs'][j]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_T # 单位tesla
            raw.info['chs'][j]['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2 # Qusqpin类型
            raw.info['chs'][j]['loc'][3:12] = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.]) # 旋转矩阵
            Z_orient =  mne._fiff.tag._loc_to_coil_trans(raw.info['chs'][j]['loc'])[:3, :3]
            find_Rotation = mne.transforms._find_vector_rotation(Z_orient[:, 2], ori[j, :])
            raw.info['chs'][j]['loc'][3:12] = np.dot(find_Rotation, Z_orient).T.ravel()

    meg_trigger = np.loadtxt(trigger_path,dtype='int')

    raw_meg = raw.copy()
    events = mne.find_events(raw_meg,stim_channel='Trigger')
    inter_trial = np.diff(events[:,0])

    # raw_meg.plot(events=events, event_color='red')

    sfreq = 1000
    trigger_data = raw_meg._data[-1]
    for i,it in enumerate(inter_trial):
        if it < 4 * sfreq:
            trigger_data[events[i+1,0]-100:events[i+1,0]+100] = 0

    raw_meg._data[-1] = trigger_data

    events = mne.find_events(raw_meg,stim_channel='Trigger')
    inter_trial = np.diff(events[:,0])

    inter_trial_pick = [it for it in inter_trial if it < 7 * sfreq]
    mean_inter_trial = int(np.mean(inter_trial_pick))

    for i,it in enumerate(inter_trial):
        if it > 8 * sfreq and it < 15 * sfreq:
            trigger_data[events[i,0]+mean_inter_trial:events[i,0]+mean_inter_trial+10] = 1

    raw_meg._data[-1] = trigger_data

    events = mne.find_events(raw_meg,stim_channel='Trigger')
    inter_trial = np.diff(events[:,0])
    # raw_meg.plot(events=events, event_color='red')

    skip_idx = []
    for i,it in enumerate(inter_trial):
        if it > 8 * sfreq and it < 15 * sfreq:
            skip_idx.append(i+len(skip_idx)+1)
        if it > 40 * sfreq:
            skip_idx.append(i+len(skip_idx)+1)
        if it > 15 * sfreq and it < 20 * sfreq:
            skip_idx.append(i+len(skip_idx)+1)
            skip_idx.append(i+len(skip_idx)+1)

    num_events = meg_trigger.shape[0]
    j = 0
    for i in range(num_events):
        if i in skip_idx:
            continue
        else:
            events[j,2] = meg_trigger[i,1]
            j += 1
    
    return raw_meg, events


def read_room_data(data_path,sensor_path):
    fs = 1000  # 采样率1000Hz
    n_record_chans = 66 # 记录通道数
    file_id = open(data_path, "rb")  # 二进制文件
    baseDate_data = np.fromfile(file_id, dtype=np.float32)  # 读取二进制文件
    General_Time_In_Seconds = len(baseDate_data) // n_record_chans // fs  # 读取数据的轮数
    Single_Sensor_Data_Length = General_Time_In_Seconds * fs  # 单探头的数据量
    file_id.close()  # 关闭文件
    read_raw_data = np.zeros((n_record_chans, Single_Sensor_Data_Length))  # 创建一个空数组，用于存放拼接后的数据
    for channel_index in range(n_record_chans):  # 遍历探头（通道数）
        for time_seconds in range(General_Time_In_Seconds):  # 遍历记录轮数(每秒的数据）
            # 将baseDate_data中对应位置的数据赋值给All_Channel_Data中对应位置
            read_raw_data[channel_index, time_seconds * fs:(time_seconds + 1) * fs] = baseDate_data[channel_index * fs + (time_seconds * n_record_chans * fs):(channel_index + 1) * fs + (time_seconds * n_record_chans * fs)]

    use_chans = 65
    raw_data = read_raw_data[:use_chans, :]
    raw_data[:-1, :] = raw_data[:-1, :] * 1e-12

    num_chans_data = 64
    sensor_info = scio.loadmat(sensor_path)

    label = list(sensor_info['ch_names'])
    label = [lab.strip() for lab in label]
    pos = sensor_info['pos']
    ori = sensor_info['ori']

    sfreq = 1000
    num_chan = 64
    raw_info = mne.create_info(
    ch_names = label + ['Trigger'],
    ch_types = ['eeg' for i in range(num_chan)] + ['stim'],sfreq=sfreq)
    raw = mne.io.RawArray(raw_data, raw_info)

    dic = {label[i]: pos[i,:] for i in range(num_chans_data)}
    montage = mne.channels.make_dig_montage(ch_pos=dic, coord_frame='head')
    raw = raw.set_montage(montage)

    for j,ch_name in enumerate(raw.info['ch_names']):
        if ch_name != 'Trigger':
            raw.info['chs'][j]['kind'] = mne.io.constants.FIFF.FIFFV_MEG_CH # 通道类型
            raw.info['chs'][j]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_T # 单位tesla
            raw.info['chs'][j]['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2 # Qusqpin类型
            raw.info['chs'][j]['loc'][3:12] = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.]) # 旋转矩阵
            Z_orient =  mne._fiff.tag._loc_to_coil_trans(raw.info['chs'][j]['loc'])[:3, :3]
            find_Rotation = mne.transforms._find_vector_rotation(Z_orient[:, 2], ori[j, :])
            raw.info['chs'][j]['loc'][3:12] = np.dot(find_Rotation, Z_orient).T.ravel()
    return raw

def plot_eeg_sensors(info, subject, subjects_dir,trans=None,masks=None,color=None):
    if trans is None:
        trans = mne.transforms.Transform('head', 'mri')
    fig = mne.viz.create_3d_figure((400,400), bgcolor=(255, 255, 255))
    # 绘制大脑
    brain=mne.viz.Brain(subject, hemi="both", surf="pial", subjects_dir=subjects_dir,
                        units='m',alpha=0.8,figure=fig,cortex=(0.7,0.7,0.7))
    brain.add_label('V1_exvivo.thresh',color='Orange', alpha=1,borders=False,hemi='lh')
    brain.add_label('V1_exvivo.thresh',color='Orange', alpha=1,borders=False,hemi='rh')

    brain.add_label('V2_exvivo.thresh',color='cyan', alpha=1,borders=False,hemi='lh')
    brain.add_label('V2_exvivo.thresh',color='cyan', alpha=1,borders=False,hemi='rh')

    # 绘制头皮表面
    coord_frame = 'mri'
    meg = ['sensors']
    bem = None
    head_surface = ["head-dense"]
    fig=wfl_plot_alignment(
        info,
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
        surfaces=[],
        eeg=[],
        meg=meg,
        coord_frame=coord_frame,
        helmet_alpha=1,helmet_color='silver',
        head_surface = head_surface, head_alpha=0.5,head_color='#C0C0C0',
        ch_pos=False,pos_scale=0.005,pos_color='darkorange',
        ch_ori=False,ori_color='#F6776E',
        fig=fig
    )

    # 绘制EEG电极
    from mne.viz.backends.renderer import _get_renderer
    from mne.viz._3d import _handle_sensor_types
    from mne.transforms import _get_trans,_get_transforms_to_coord_frame
    from mne.bem import ConductorModel, _ensure_bem_surfaces
    from collections import defaultdict
    from mne._fiff.pick import (
        _FNIRS_CH_TYPES_SPLIT,
        _MEG_CH_TYPES_SPLIT,
        channel_type,
        pick_info,
        pick_types,
    )
    from mne.surface import _project_onto_surface
    from mne.viz._3d import _ch_pos_in_coord_frame
    from mne.defaults import DEFAULTS

    renderer = _get_renderer(fig)

    eeg = ['original','projected']
    head_surface = ["head-dense"]
    fnirs = []
    units = 'm'

    picks = pick_types(info,meg=("sensors" in meg),ref_meg=("ref" in meg),
                    eeg=(len(eeg) > 0),ecog=False,seeg=False,dbs=False,
                    fnirs=(len(fnirs) > 0))

    meg, eeg, fnirs, warn_meg, sensor_alpha = _handle_sensor_types(meg, eeg, fnirs)

    trans, trans_type = _get_trans(trans, fro="head", to="mri")
    head_mri_t = _get_trans(trans, "head", "mri")[0]
    to_cf_t = _get_transforms_to_coord_frame(info, head_mri_t, coord_frame=coord_frame)

    # Head surface:
    head_keys = ("auto", "head", "outer_skin", "head-dense", "seghead")
    head = [s for s in head_surface if s in head_keys]
    if len(head) > 1:
        raise ValueError("Can only supply one head-like surface name, " f"got {head}")
    head = head[0] if head else False
    if head is not False:
        head_surface.pop(head_surface.index(head))
    elif "projected" in eeg:
        raise ValueError(
            "A head surface is required to project EEG, "
            '"head", "outer_skin", "head-dense" or "seghead" '
            'must be in surfaces or surfaces must be "auto"'
        )
    bem = _ensure_bem_surfaces(bem, extra_allow=(ConductorModel, None))
    assert isinstance(bem, ConductorModel) or bem is None

    from mne._freesurfer import _get_head_surface
    from mne.transforms import transform_surface_to
    head_surf = _get_head_surface(head, subject, subjects_dir, bem=bem)
    head_surf = transform_surface_to(
        head_surf, coord_frame, [to_cf_t["mri"], to_cf_t["head"]], copy=True
    )

    """Render sensors in a 3D scene."""
    ch_pos, sources, detectors = _ch_pos_in_coord_frame(
        pick_info(info, picks), to_cf_t=to_cf_t, warn_meg=warn_meg)

    locs = defaultdict(lambda: list())
    unit_scalar = 1 if units == "m" else 1e3
    for ch_name, ch_coord in ch_pos.items():
        ch_type = channel_type(info, info.ch_names.index(ch_name))
        # for default picking
        if ch_type in _FNIRS_CH_TYPES_SPLIT:
            ch_type = "fnirs"
        elif ch_type in _MEG_CH_TYPES_SPLIT:
            ch_type = "meg"
        # only plot sensor locations if channels/original in selection
        plot_sensors = (ch_type != "fnirs" or "channels" in fnirs) and (
            ch_type != "eeg" or "original" in eeg
        )
        # plot sensors
        if isinstance(ch_coord, tuple):  # is meg, plot coil
            ch_coord = dict(rr=ch_coord[0] * unit_scalar, tris=ch_coord[1])
        if plot_sensors:
            locs[ch_type].append(ch_coord)
        if ch_name in sources and "sources" in fnirs:
            locs["source"].append(sources[ch_name])
        if ch_name in detectors and "detectors" in fnirs:
            locs["detector"].append(detectors[ch_name])

    for ch_type, sens_loc in locs.items():
        print(f"Drawing {ch_type} sensors")
        assert len(sens_loc)  # should be guaranteed above

        sens_loc = np.array(sens_loc, float)
        num_chans = sens_loc.shape[0]
        if masks is None:
            masks = ['P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO3',
                     'POZ','PO4','PO8','O1','OZ','O2']
        picks = _picks_to_idx(info,picks=masks,exclude=[])
        mask1 = np.array([False if i in picks else True for i in range(num_chans) ])
        mask2 = np.array([True if i in picks else False for i in range(num_chans) ])

        colors = ['#1E90FF','Crimson']
        if color is not None:
            colors = color

        if ch_type == "eeg" and "projected" in eeg:
            print("Projecting sensors to the head surface")
            for mask,color in zip([mask1,mask2],colors):
                eegp_loc, eegp_nn = _project_onto_surface(
                    sens_loc[mask], head_surf, project_rrs=True, return_nn=True
                )[2:4]
                eegp_loc *= unit_scalar
                actor, _ = renderer.quiver3d(
                    x=eegp_loc[:, 0],
                    y=eegp_loc[:, 1],
                    z=eegp_loc[:, 2],
                    u=eegp_nn[:, 0],
                    v=eegp_nn[:, 1],
                    w=eegp_nn[:, 2],
                    color=color,
                    mode="cylinder",
                    # scale=defaults["eegp_scale"] * unit_scalar,
                    scale=0.03,
                    opacity=sensor_alpha["eeg_projected"],
                    # glyph_height=defaults["eegp_height"],
                    glyph_height=0.03,
                    glyph_center=(0.0, -0.03 / 2.0, 0),
                    glyph_resolution=20,
                    backface_culling=True,
                    glyph_radius=0.15,
                )

    mne.viz.set_3d_view(fig, azimuth=-90, elevation=75,roll=2,focalpoint='auto', distance='auto')
    return fig


def plot_meg_sensors(info,subject,subjects_dir,trans=None,masks=None,color=None):
    fig = mne.viz.create_3d_figure((400,400), bgcolor=(255, 255, 255))

    # 绘制大脑
    brain=mne.viz.Brain(subject, hemi="both", surf="pial", subjects_dir=subjects_dir,
                        units='m',alpha=0.8,figure=fig,cortex=(0.7,0.7,0.7))

    brain.add_label('V1_exvivo.thresh',color='Orange', alpha=1,borders=False,hemi='lh')
    brain.add_label('V1_exvivo.thresh',color='Orange', alpha=1,borders=False,hemi='rh')

    brain.add_label('V2_exvivo.thresh',color='cyan', alpha=1,borders=False,hemi='lh')
    brain.add_label('V2_exvivo.thresh',color='cyan', alpha=1,borders=False,hemi='rh')

    coord_frame = 'mri'
    fig=wfl_plot_alignment(
        info,
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
        surfaces=[],
        show_axes=False,
        dig=False,
        eeg=[],
        meg=["sensors"],
        coord_frame=coord_frame,
        helmet_alpha=1,helmet_color='silver',
        head_surface = ["head-dense"], head_alpha=0.5,head_color='#C0C0C0',
        ch_pos=False,pos_scale=0.005,pos_color='blue',
        ch_ori=False,ori_color='#F6776E',
        fig=fig
    )

    # 绘制EEG电极
    renderer = _get_renderer(fig)
    trans, trans_type = _get_trans(trans, fro="head", to="mri")
    head_mri_t = _get_trans(trans, "head", "mri")[0]
    to_cf_t = _get_transforms_to_coord_frame(info, head_mri_t, coord_frame=coord_frame)


    chs = info["chs"]
    pos_meg = []
    ori_meg = []
    for ci, ch in enumerate(chs):
        pos1 = ch["loc"][:3]
        if ch["coord_frame"] != mne.io.constants.FIFF.FIFFV_COORD_UNKNOWN:
            pos_meg.append(apply_trans(to_cf_t['meg'], pos1))
            ori_meg.append(ch["loc"][9:12])
    pos_meg = np.array(pos_meg)
    ori_meg = np.array(ori_meg)

    if masks is None:
        masks = ['P7','P5','P3','P1','PZ','P2','P4','P6','P8','P9','P10',
                        'PO7','PO3','POZ','PO4','PO8','O1','OZ','O2','IZ']
    picks = _picks_to_idx(info,picks=masks,exclude=[])

    mask1 = np.array([False if i in picks else True for i in range(64)])
    mask2 = np.array([True if i in picks else False for i in range(64)])
    colors = ['#1E90FF','Crimson']
    if color is not None:
        colors = color
        
    for mask,color in zip([mask1,mask2],colors):
        pos_pick = pos_meg[mask]
        ori_pick = ori_meg[mask]
        actor, _ = quiver3d(renderer,
            x=pos_pick[:, 0],
            y=pos_pick[:, 1],
            z=pos_pick[:, 2],
            u=ori_pick[:, 0],
            v=ori_pick[:, 1],
            w=ori_pick[:, 2],
            color=color,
            mode="cube",
            scale=0.03,
            opacity=1,
            glyph_center=(0.0, -0.03 / 2.0, 0),
            glyph_lengths = (0.3,0.3,0.05),
            backface_culling=True,
            glyph_radius=0.15,
        )

    mne.viz.set_3d_view(fig, azimuth=-90, elevation=80,roll=2,focalpoint='auto', distance='auto')
    return fig


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
                     mask=None,mask_params=None,text_args=None,scatter_args=None):
    
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

    scatter_args = dict() if scatter_args is None else scatter_args.copy()
    s = scatter_args.get("s", None)
    facecolors = scatter_args.get("facecolors", 'k')
    marker = scatter_args.get("marker", "o")
    linewidths = scatter_args.get("linewidths", 0.5)
    alpha = scatter_args.get("alpha", 1.0)
    edgecolors = scatter_args.get("edgecolors", None)
    plotnonfinite = scatter_args.get("plotnonfinite", False)

    if mask is not None:
        picks = _picks_to_idx(info,picks=mask,exclude=[])
        mask1 = np.array([False if i in picks else True for i in range(num_chans) ])
        mask2 = np.array([True if i in picks else False for i in range(num_chans) ])
        pos1 = pos[mask1]
        pos2 = pos[mask2]
        show_names1 = [ch_names[i] for i,b in enumerate(mask1) if b]
        show_names2 = [ch_names[i] for i,b in enumerate(mask2) if b]


    if mask_params is None:
        mask_params = dict(marker='o', markerfacecolor='r', markeredgecolor='k',
                           linewidth=1, markersize=11,markeralpha=1,
                           textcolor='r',textalpha=1)
        mask_params = {'marker':'o', 'markerfacecolor':'r', 'markeredgecolor':'r',
                       'markersize':11,'markeralpha':1,'linewidth':1,
                       'textcolor':'r','textalpha':1}
    if mask is None:
        pts = ax.scatter(pos[:, 0],pos[:, 1],picker=True,clip_on=False,marker=marker,
                        c=None,s=s,linewidths=linewidths,alpha=alpha,
                        facecolors=facecolors, edgecolors=edgecolors,
                        plotnonfinite=plotnonfinite)
        if show_names:
            if isinstance(show_names, (list, np.ndarray)):  # only given channels
                indices = [list(ch_names).index(name) for name in show_names]
            else:  # all channels
                indices = range(len(pos))
            text_args = dict() if text_args is None else text_args.copy()
            alpha = text_args.get("alpha", 1.0)
            color = text_args.get("color", "k")
            fontsize = text_args.get("fontsize", None)
            fontstyle = text_args.get("fontstyle", None)
            fontweight = text_args.get("fontweight", None)

            for idx in indices:
                this_pos = pos[idx]
                ax.text(this_pos[0],this_pos[1],ch_names[idx],
                    ha="center",va="center",
                    alpha=alpha,color=color,fontsize=fontsize,
                    fontweight=fontweight,fontstyle=fontstyle
                )
    else:
        markers = [marker, mask_params.get("marker", marker)]
        marker_ss = [s, mask_params.get("markersize", s)]
        marker_facecolorss = [facecolors, mask_params.get("markerfacecolor", facecolors)]
        marker_edgecolorss = [edgecolors, mask_params.get("markeredgecolor", edgecolors)]
        marker_linewidthss = [linewidths, mask_params.get("linewidth", linewidths)]
        marker_alphas = [alpha, mask_params.get("markeralpha", alpha)]

        text_alphas = [text_args.get("alpha", 1.0),
                 mask_params.get("textalpha", text_args.get("alpha", 1.0))]
        text_colors = [text_args.get("color", "k"),
                 mask_params.get("textcolor", text_args.get("color", "k"))]
        text_fontsizes = [text_args.get("fontsize", None),
                    mask_params.get("fontsize", text_args.get("fontsize", None))]
        text_fontstyles = [text_args.get("fontstyle", None),
                     mask_params.get("fontstyle", text_args.get("fontstyle", None))]
        text_fontweights = [text_args.get("fontweight", None),
                      mask_params.get("fontweight", text_args.get("fontweight", None))]


        for i,(pos,show_names) in enumerate(zip([pos1, pos2], [show_names1, show_names2])):
            marker = markers[i]
            marker_s = marker_ss[i]
            marker_facecolors = marker_facecolorss[i]
            marker_edgecolors = marker_edgecolorss[i]
            marker_linewidths = marker_linewidthss[i]
            marker_alpha = marker_alphas[i]
            
            pts = ax.scatter(pos[:, 0],pos[:, 1],picker=True,clip_on=False,marker=marker,
                            c=None,s=marker_s,linewidths=marker_linewidths,alpha=marker_alpha,
                            facecolors=marker_facecolors, edgecolors=marker_edgecolors,
                            plotnonfinite=plotnonfinite)
            if show_names:
                indices = range(len(pos))

                text_alpha = text_alphas[i]
                text_color = text_colors[i]
                text_fontsize = text_fontsizes[i]
                text_fontstyle = text_fontstyles[i]
                text_fontweight = text_fontweights[i]
                for idx in indices:
                    this_pos = pos[idx]
                    ax.text(this_pos[0] + 0.0025,this_pos[1],show_names[idx],
                        ha="left",va="center",
                        alpha=text_alpha,color=text_color,fontsize=text_fontsize,
                        fontweight=text_fontweight,fontstyle=text_fontstyle
                    )
                    # ax.text(this_pos[0],this_pos[1],show_names[idx],
                    #     ha="center",va="center",
                    #     alpha=text_alpha,color=text_color,fontsize=text_fontsize,
                    #     fontweight=text_fontweight,fontstyle=text_fontstyle
                    # )
    

    ax.set(aspect="equal")
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR, vtkCommand, vtkLookupTable
from vtkmodules.vtkCommonDataModel import VTK_VERTEX, vtkPiecewiseFunction
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkGlyph3D
from vtkmodules.vtkFiltersGeneral import (
    vtkMarchingContourFilter,
    vtkTransformPolyDataFilter,
)
from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
from vtkmodules.vtkFiltersSources import (
    vtkArrowSource,
    vtkConeSource,
    vtkCylinderSource,
    vtkGlyphSource2D,
    vtkPlatonicSolidSource,
    vtkSphereSource,
    vtkCubeSource,
)
from vtkmodules.vtkImagingCore import vtkImageReslice
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkColorTransferFunction,
    vtkCoordinate,
    vtkDataSetMapper,
    vtkMapper,
    vtkPolyDataMapper,
    vtkVolume,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

import pyvista
from pyvista import Line, Plotter, PolyData, UnstructuredGrid, close_all

def quiver3d(
        my_render,
        x,
        y,
        z,
        u,
        v,
        w,
        color,
        scale,
        mode,
        resolution=8,
        *,
        glyph_height=None,
        glyph_center=None,
        glyph_resolution=None,
        glyph_lengths = None,
        opacity=1.0,
        scale_mode="none",
        scalars=None,
        colormap=None,
        backface_culling=False,
        glyph_radius=0.15,
        solid_transform=None,
        clim=None,
    ):

        scale_map = dict(none=False, scalar="scalars", vector="vec")
        factor = scale
        vectors = np.c_[u, v, w]
        points = np.vstack(np.c_[x, y, z])
        n_points = len(points)
        cell_type = np.full(n_points, VTK_VERTEX)
        cells = np.c_[np.full(n_points, 1), range(n_points)]
        args = (cells, cell_type, points)
        grid = UnstructuredGrid(*args)
        if scalars is None:
            scalars = np.ones((n_points,))
            mesh_scalars = None
        else:
            mesh_scalars = "scalars"
        grid.point_data["scalars"] = np.array(scalars, float)
        grid.point_data["vec"] = vectors

        if mode == "cone":
            glyph = vtkConeSource()
            glyph.SetCenter(0.5, 0, 0)
            if glyph_radius is not None:
                glyph.SetRadius(glyph_radius)
        elif mode == "cylinder":
            glyph = vtkCylinderSource()
            if glyph_radius is not None:
                glyph.SetRadius(glyph_radius)
        elif mode == "oct":
            glyph = vtkPlatonicSolidSource()
            glyph.SetSolidTypeToOctahedron()
        elif mode == "cube":
            glyph = vtkCubeSource()
        else:
            assert mode == "sphere", mode  # guaranteed above
            glyph = vtkSphereSource()
        if mode == "cylinder":
            if glyph_height is not None:
                glyph.SetHeight(glyph_height)
            if glyph_center is not None:
                glyph.SetCenter(glyph_center)
            if glyph_resolution is not None:
                glyph.SetResolution(glyph_resolution)
            tr = vtkTransform()
            tr.RotateWXYZ(90, 0, 0, 1)
        elif mode == "oct":
            if solid_transform is not None:
                assert solid_transform.shape == (4, 4)
                tr = vtkTransform()
                tr.SetMatrix(solid_transform.astype(np.float64).ravel())
        if mode == 'cube':
            if glyph_lengths is not None:
                glyph.SetXLength(glyph_lengths[0])
                glyph.SetYLength(glyph_lengths[2])
                glyph.SetZLength(glyph_lengths[1])
            if glyph_center is not None:
                glyph.SetCenter(glyph_center)
            tr = vtkTransform()
            tr.RotateWXYZ(90, 0, 0, 1)
        if tr is not None:
            # fix orientation
            glyph.Update()
            trp = vtkTransformPolyDataFilter()
            trp.SetInputData(glyph.GetOutput())
            trp.SetTransform(tr)
            glyph = trp
        glyph.Update()
        geom = glyph.GetOutput()
        mesh = grid.glyph(
            orient="vec",
            scale=scale_map[scale_mode],
            factor=factor,
            geom=geom,
        )
        actor = _add_mesh(
            my_render.plotter,
            mesh=mesh,
            color=color,
            opacity=opacity,
            scalars=mesh_scalars if colormap is not None else None,
            colormap=colormap,
            show_scalar_bar=False,
            backface_culling=backface_culling,
            clim=clim,
        )
        return actor, mesh

def _add_mesh(plotter, *args, **kwargs):
    """Patch PyVista add_mesh."""
    mesh = kwargs.get("mesh")
    if "smooth_shading" in kwargs:
        smooth_shading = kwargs.pop("smooth_shading")
    else:
        smooth_shading = True
    if "render" not in kwargs:
        kwargs["render"] = False
    if "reset_camera" not in kwargs:
        kwargs["reset_camera"] = False
    actor = plotter.add_mesh(*args, **kwargs)
    if smooth_shading and "Normals" in mesh.point_data:
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
    return actor




