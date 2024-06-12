"""
Create LMDB for the training set of VCP. Compressed video with prior dataset 

GT: 300 training sequences out of 300 48-frame sequences.
LQ: HM16.5-intra-compressed sequences.
key: assigned from 000 to 048.

Sym-link VCP dataset root to ./data/VCP folder.
"""
import argparse
import os
import glob
import yaml
import os.path as op
from utils import make_lmdb_from_imgs, make_lmdb_from_npys

parser = argparse.ArgumentParser()
parser.add_argument(
    '--opt_path', type=str, default='option_R3_cvpd_LDB_37.yml', help='Path to option YAML file.')
args = parser.parse_args()

yml_path = args.opt_path
radius = 3  # must be 3!!! otherwise, you should change dataset.py


def create_lmdb_for_vcp():
    # video info
    with open(yml_path, 'r') as fp:
        fp = yaml.load(fp, Loader=yaml.FullLoader)
        root_dir = fp['dataset']['root']
        gt_folder = fp['dataset']['train']['gt_folder']
        lq_folder = fp['dataset']['train']['lq_folder']
        mv_folder = fp['dataset']['train']['mv_folder']
        pm_folder = fp['dataset']['train']['pm_folder']
        rm_folder = fp['dataset']['train']['rm_folder']
        pd_folder = fp['dataset']['train']['pd_folder']
        gt_path = fp['dataset']['train']['gt_path']
        lq_path = fp['dataset']['train']['lq_path']
        mv_path = fp['dataset']['train']['mv_path']
        pm_path = fp['dataset']['train']['pm_path']
        rm_path = fp['dataset']['train']['rm_path']
        pd_path = fp['dataset']['train']['pd_path']
        gtmeta_path = fp['dataset']['train']['gtmeta_path']
        lqmeta_path = fp['dataset']['train']['lqmeta_path']
    gt_dir = op.join(root_dir, gt_folder)
    lq_dir = op.join(root_dir, lq_folder)
    lmdb_gt_path = op.join(root_dir, gt_path)
    lmdb_lq_path = op.join(root_dir, lq_path)
    lmdb_mv_path = op.join(root_dir, mv_path)
    lmdb_pm_path = op.join(root_dir, pm_path)
    lmdb_rm_path = op.join(root_dir, rm_path)
    lmdb_pd_path = op.join(root_dir, pd_path)
    GTmeta_path = op.join(root_dir, gtmeta_path)
    LQmeta_path = op.join(root_dir, lqmeta_path)

    # scan all videos
    print('Scaning meta list...')
    gt_video_list = []
    lq_video_list = []
    mv_video_list = []
    pm_video_list = []
    rm_video_list = []
    pd_video_list = []
    # meta_fp = open(GTmeta_path, 'r')
    meta_fp = open(LQmeta_path, 'r')
    while True:
        new_line = meta_fp.readline().split('\n')[0]
        if new_line == '':
            break
        vid_name = new_line
        gt_path = op.join( gt_dir, vid_name)
        lq_path = op.join( lq_dir, vid_name)
        mv_path = op.join( lq_dir, vid_name.replace('compress','mv'))
        pm_path = op.join( lq_dir, vid_name.replace('compress','PAI'))
        rm_path = op.join( lq_dir, vid_name.replace('compress','residue'))
        pd_path = op.join( lq_dir, vid_name.replace('compress','pred'))
        gt_video_list.append(gt_path)
        lq_video_list.append(lq_path)
        mv_video_list.append(mv_path) 
        pm_video_list.append(pm_path) 
        rm_video_list.append(rm_path) 
        pd_video_list.append(pd_path)        

    msg = f'> {len(gt_video_list)} videos found.'
    print(msg)

    # generate LMDB for GT
    print("Scaning GT frames (only center frames of each sequence)...")
    frm_list = []
    for gt_video_path in gt_video_list:
        nfs = 48 # float(gt_video_path.split('/')[0].split('_')[-1])
        num_seq = nfs // (2 * radius + 1)
        frm_list.append([radius + iter_seq * (2 * radius + 1) + 1 for iter_seq in range(num_seq)])
    num_frm_total = sum([len(frms) for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)
    keys = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(gt_video_list)):
        frms = frm_list[iter_vid]
        for iter_frm in range(len(frms)):
            keys.append('{:03d}/{:03d}/004.png'.format(iter_vid+1, iter_frm+1))  #  004  008
            video_path_list.append(gt_video_list[iter_vid])
            index_frame_list.append(frms[iter_frm])
    print("Writing LMDB for GT data...")
    # make_lmdb_from_imgs(img_dir=video_path_list,lmdb_path=lmdb_gt_path, img_path_list=video_path_list, index_frame_list=index_frame_list, keys=keys, \
    #     batch=5000, compress_level=0,  multiprocessing_read=True, map_size=None)
    print("> Finish.")

    # generate LMDB for LQ
    print("Scaning LQ frames...")
    len_input = 2 * radius + 1
    frm_list = []
    for lq_video_path in lq_video_list:
        nfs = 48 
        num_seq = nfs // len_input
        frm_list.append([list(range(iter_seq * len_input + 1, (iter_seq + 1) * len_input + 1)) for iter_seq in range(num_seq)])
    num_frm_total = sum([len(frms) * len_input for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)
    keys = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(lq_video_list)):
        frm_seq = frm_list[iter_vid]
        for iter_seq in range(len(frm_seq)):
            # keys.extend(['{}/{:03d}.png'.format(lq_video_list[iter_vid], i) for i in range(1, len_input+1)])
            keys.extend(['{:03d}/{:03d}/{:03d}.png'.format(iter_vid+1, iter_seq+1, i) for i in range(1, len_input+1)])
            video_path_list.extend([lq_video_list[iter_vid]] * len_input)  #  [lq_video_list[iter_vid]] * len_input
            index_frame_list.extend(frm_seq[iter_seq])
            
    print("Writing LMDB for LQ data...")
    # make_lmdb_from_imgs(img_dir=video_path_list, lmdb_path=lmdb_lq_path, img_path_list=video_path_list, index_frame_list=index_frame_list, keys=keys, \
    #     batch=5000, compress_level=0, multiprocessing_read=True, map_size=None)
    print("> Finish.")


    # generate LMDB for PAI
    print("Scaning PAI frames...")
    len_input = 2 * radius + 1
    frm_list = []
    for pm_video_path in pm_video_list:
        nfs = 48 
        num_seq = nfs // len_input
        frm_list.append([list(range(iter_seq * len_input+ 1, (iter_seq + 1) * len_input+ 1)) for iter_seq in range(num_seq)])
    num_frm_total = sum([len(frms) * len_input for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)
    keys = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(pm_video_list)):
        frm_seq = frm_list[iter_vid]
        for iter_seq in range(len(frm_seq)):
            # keys.extend(['{}/PAI_{:03d}.png'.format(pm_video_list[iter_vid], i) for i in range(1, len_input+1)])
            keys.extend(['{:03d}/{:03d}/{:03d}.png'.format(iter_vid+1, iter_seq+1, i) for i in range(1, len_input+1)])
            video_path_list.extend([pm_video_list[iter_vid]] * len_input)  
            index_frame_list.extend(frm_seq[iter_seq])
            
    print("Writing LMDB for PAI data...")
    # make_lmdb_from_imgs(img_dir=video_path_list, lmdb_path=lmdb_pm_path, img_path_list=video_path_list,index_frame_list=index_frame_list, keys=keys, \
    #     batch=5000, compress_level=0, multiprocessing_read=True, map_size=None)
    print("> Finish.")

    # generate LMDB for Pred
    print("Scaning Pred frames...")
    len_input = 2 * radius + 1
    frm_list = []
    for pd_video_path in pd_video_list:
        nfs = 48 
        num_seq = nfs // len_input
        frm_list.append([list(range(iter_seq * len_input+1, (iter_seq + 1) * len_input+1)) for iter_seq in range(num_seq)])
    num_frm_total = sum([len(frms) * len_input for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)
    keys = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(pd_video_list)):
        frm_seq = frm_list[iter_vid]
        for iter_seq in range(len(frm_seq)):
            keys.extend(['{:03d}/{:03d}/{:03d}.png'.format(iter_vid+1, iter_seq+1, i) for i in range(1, len_input+1)])
            video_path_list.extend([pd_video_list[iter_vid]] * len_input)  
            index_frame_list.extend(frm_seq[iter_seq])
            
    print("Writing LMDB for Pred data...")
    # make_lmdb_from_imgs(img_dir=video_path_list, lmdb_path=lmdb_pd_path, img_path_list=video_path_list, index_frame_list=index_frame_list, keys=keys, \
    #     batch=5000, compress_level=0, multiprocessing_read=True, map_size=None)
    print("> Finish.")

    # generate LMDB for MV
    print("Scaning MVs...")
    len_input = 2 * radius + 1
    frm_list = []
    for mv_video_path in mv_video_list:
        nfs = 48 
        num_seq = nfs // len_input
        frm_list.append([list(range(iter_seq * len_input+1, (iter_seq + 1) * len_input+1)) for iter_seq in range(num_seq)])
    num_frm_total = sum([len(frms) * len_input for frms in frm_list])
    msg = f'> {num_frm_total} npys found.'
    print(msg)
    keys = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(mv_video_list)):
        frm_seq = frm_list[iter_vid]
        for iter_seq in range(len(frm_seq)):
            keys.extend(['{:03d}/{:03d}/{:03d}.npy'.format(iter_vid+1, iter_seq+1, i) for i in range(1, len_input+1)])
            video_path_list.extend([mv_video_list[iter_vid]] * len_input)
            index_frame_list.extend(frm_seq[iter_seq])
    print('[keys]',keys[0],len(video_path_list))
    print("Writing LMDB for MV data...")
    make_lmdb_from_npys(npy_dir=video_path_list, lmdb_path=lmdb_mv_path, npy_path_list=video_path_list, index_frame_list=index_frame_list, keys=keys, \
        batch=5000, compress_level=0, multiprocessing_read=False, map_size=None)
    print("> Finish.")


    # generate LMDB for Residual

    # keys = []
    # video_path_list = []
    # index_frame_list = []
    # for iter_vid in range(len(gt_video_list)):
    #     frms = frm_list[iter_vid]
    #     for iter_frm in range(len(frms)):
    #         keys.append('{:03d}/{:03d}/008.png'.format(iter_vid+1, iter_frm+1))  #  004  008
    #         video_path_list.append(gt_video_list[iter_vid])
    #         index_frame_list.append(frms[iter_frm])


    print("Scaning Residuals...")
    len_input = 2 * radius + 1
    frm_list = []

    for rm_video_path in rm_video_list:
        nfs = 48 
        num_seq = nfs // len_input
        # frm_list.append([radius + iter_seq * (2 * radius + 1) + 1 for iter_seq in range(num_seq)])
        frm_list.append([list(range(iter_seq * len_input+1 , (iter_seq + 1) * len_input+1)) for iter_seq in range(num_seq)])
    num_frm_total = sum([len(frms) * len_input for frms in frm_list])
    # num_frm_total = sum([len(frms) for frms in frm_list])
    msg = f'> {num_frm_total} npys found.'
    print(msg)
    keys = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(rm_video_list)):
        frm_seq = frm_list[iter_vid]
        for iter_seq in range(len(frm_seq)):
            keys.append('{:03d}/{:03d}/004.npy'.format(iter_vid+1, iter_seq+1))  #  004  008
            # keys.extend(['{:03d}/{:03d}/{:03d}.npy'.format(iter_vid+1, iter_seq+1, i) for i in range(1, len_input+1)])
            # keys.append('{:03d}/{:03d}/008.png'.format(iter_vid+1, iter_frm+1))  #  004  008
            video_path_list.extend(rm_video_list[iter_vid])
            index_frame_list.extend(frm_seq[iter_seq])
    print("Writing LMDB for Residuals data...")
    # make_lmdb_from_npys(npy_dir=video_path_list, lmdb_path=lmdb_rm_path, npy_path_list=video_path_list, index_frame_list=index_frame_list, keys=keys, \
    #     batch=5000, compress_level=0, multiprocessing_read=True, map_size=None)
    print("> Finish.")

    # # sym-link
    # if not op.exists('data/cvpd'):
    #     if not op.exists('data/'):
    #         os.system("mkdir data/")
    #     os.system(f"ln -s {root_dir} ./data/cvpd")
    #     print("Sym-linking done.")
    # else:
    #     print("data/vcp already exists.")
    




if __name__ == '__main__':
    create_lmdb_for_vcp()

