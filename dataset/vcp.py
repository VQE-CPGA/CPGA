import glob
import random
import torch
import os.path as op
import numpy as np
import os
# from cv2 import cv2
import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, paired_random_crop_prior, augment, totensor, import_yuv, paired_random_crop_prior_0, paired_random_crop_prior_0res, paired_random_crop_prior_res, paired_random_crop_prior_mv, paired_random_crop_prior_mvpred
import torch.nn.functional as F

def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    # print('[img_bytes]',img_np.shape)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img


def _bytes2MVnpy(npy_bytes, h, w, c):
    img = np.frombuffer(npy_bytes, dtype=np.float32)
    # img = np.frombuffer(npy_bytes, np.int8)
    img_np = img.reshape(h, w, 2)
    # img_np = np.decode(img_np)
    # img_np = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)
    # img = np.expand_dims(img_np, 2)  # (H W 1)
    img = img_np.astype(np.float32)
    return img_np


def _bytes2MVnpyint8(npy_bytes, h, w, c):
    # img = np.frombuffer(npy_bytes, dtype=np.float32)
    img = np.frombuffer(npy_bytes, np.int8)
    img_np = img.reshape(h, w, 2)
    # img_np = np.decode(img_np)
    # img_np = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)
    # img = np.expand_dims(img_np, 2)  # (H W 1)
    # img = img_np.astype(np.float32)
    img = img_np.astype(np.int8)
    return img_np


def _bytes2MVnpyint8(npy_bytes, h, w, c):
    # img = np.frombuffer(npy_bytes, dtype=np.float32)
    img = np.frombuffer(npy_bytes, np.int8)
    img_np = img.reshape(h, w, 2)
    # img_np = np.decode(img_np)
    # img_np = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)
    # img = np.expand_dims(img_np, 2)  # (H W 1)
    # img = img_np.astype(np.float32)
    img = img_np.astype(np.int8)
    return img_np


def _bytes2Resnpy(npy_bytes, h, w, c):
    img_np = np.frombuffer(npy_bytes, dtype=np.float32)
    # img_np = np.frombuffer(npy_bytes, dtype=np.int8)
    img_np = img_np.reshape(h, w, c)
    # img_np = np.expand_dims(img_np, 2)  # (H W 1)
    img = img_np.astype(np.float32)
    # img = img_np.astype(np.int8)
    return img

def _bytes2Resnpyint8(npy_bytes, h, w, c):
    # img_np = np.frombuffer(npy_bytes, dtype=np.float32)
    img_np = np.frombuffer(npy_bytes, dtype=np.int8)
    img_np = img_np.reshape(h, w, c)
    # img_np = np.expand_dims(img_np, 2)  # (H W 1)
    # img = img_np.astype(np.float32)
    img = img_np.astype(np.int8)
    return img



class CVPDpriorDatasetV0(data.Dataset):
    """CVPD prior dataset.
    For training data: LMDB is adopted. See create_lmdb for details.
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()
        self.opts_dict = opts_dict
        # dataset paths
        self.gt_root = op.join('data/CVPD/', self.opts_dict['gt_path'])
        self.lq_root = op.join('data/CVPD/',  self.opts_dict['lq_path'])
        self.pred_root = op.join('data/CVPD/',  self.opts_dict['pd_path'])
        # self.PAI_root = op.join('data/CVPD/',  self.opts_dict['pm_path'])
        self.mv_root = op.join('data/CVPD/',  self.opts_dict['mv_path'])
        self.residue_root = op.join('data/CVPD/',  self.opts_dict['rm_path'])
        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        self.meta_info_path_res = op.join(
            self.residue_root, 
            'meta_info_center.txt'
            )
        with open(self.meta_info_path_res, 'r') as fin:
            self.keys_res = [line.split(' ')[0] for line in fin]

        # print('self.keys',len(self.keys), len(self.keys_res))

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [self.lq_root, self.gt_root, self.pred_root, self.mv_root, self.residue_root]
        self.io_opts_dict['client_keys'] = ['lq', 'gt', 'pred', 'mv', 'residue']

        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = ['{:03d}'.format(i + (9 - nfs) // 2) for i in range(nfs)]
            # self.neighbor_list = ['{:03d}'.format(i + (5 - nfs) // 2) for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)
        h, w, c = img_gt.shape

        # get the neighboring residual frames
        img_res_path = self.keys_res[index]
        img_bytes = self.file_client.get(img_res_path, 'residue')
        img_residue = _bytes2Resnpy(img_bytes, h, w, c) 
        

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/{neighbor}.png'
            # print('img_lq_path',img_lq_path)
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)

        # get the neighboring Pred frames
        img_preds = []
        for neighbor in self.neighbor_list:
            img_pred_path = f'{clip}/{seq}/{neighbor}.png'
            img_bytes = self.file_client.get(img_pred_path, 'pred')
            img_pred = _bytes2img(img_bytes)  # (H W 1)
            img_preds.append(img_pred)

        # get the neighboring PAI frames
        # img_PAIs = []
        # for neighbor in self.neighbor_list:
        #     img_PAI_path = f'{clip}/{seq}/{neighbor}.png'
        #     img_bytes = self.file_client.get(img_PAI_path, 'PAI')
        #     img_PAI = _bytes2img(img_bytes)  # (H W 1)
        #     img_PAIs.append(img_PAI)

        # get the neighboring MV frames
        img_mvs = []
        for neighbor in self.neighbor_list:
            img_pred_path = f'{clip}/{seq}/{neighbor}.png'
            img_bytes = self.file_client.get(img_pred_path, 'pred')
            img_pred = _bytes2img(img_bytes)  # (H W 1)
            h, w, c = img_pred.shape

            img_mv_path = f'{clip}/{seq}/{neighbor}.npy'
            img_bytes = self.file_client.get(img_mv_path, 'mv')
            img_mv = _bytes2MVnpyint8(img_bytes, h, w, c)  # (H W 1)  _bytes2MVnpyint8  _bytes2MVnpy
            # print('[img_mv]',img_mv.shape)
            img_mvs.append(img_mv)

        

        # ==========
        # data augmentation
        # ==========

        # randomly crop
        img_gt, img_residue, img_lqs, img_preds, img_mvs = paired_random_crop_prior_res(img_gt, img_residue, img_lqs, img_preds, img_mvs,  gt_size, img_gt_path)
        # print('img_residue',type(img_gt), type(img_residue))
        # flip, rotate
        # import ast
        # c = ast.literal_eval(b)
        for i in range(len(img_preds)):
            img_lqs.append(img_preds[i]) 
        # for i in range(len(img_PAIs)):
        #     img_lqs.append(img_PAIs[i]) 
        # for i in range(len(img_residues)):
        #     img_lqs.append(img_residues[i])
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_lqs.append(img_residue) 
        # print('img_gt',type(img_gt), type(img_residue))
        img_results, flow_results = augment(img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot'], flows=img_mvs)

        # to tensor
        img_results = totensor(img_results)
        trav_len = len(self.neighbor_list)
        img_lqs = torch.stack(img_results[0:trav_len], dim=0)
        img_preds = torch.stack(img_results[trav_len:2*trav_len], dim=0)
        # img_PAIs = torch.stack(img_results[2*trav_len:3*trav_len], dim=0)
        # img_residues = torch.stack(img_results[2*trav_len:3*trav_len], dim=0)
        img_gt = img_results[-2]
        img_residue = img_results[-1]
        img_mvs = torch.stack(totensor(flow_results), dim=0) 
        # print('[type]',(img_lqs.shape),  (img_mvs.shape), img_residue.shape, img_gt.shape)

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            # 'PAI': img_PAIs,  # (T [RGB] H W)
            'mv': img_mvs,  # (T [RGB] H W)
            'residue': img_residue,  # (T [RGB] H W)
            'pred': img_preds,  # (T [RGB] H W)
            }

    def __len__(self):
        return len(self.keys)



class VideoTestCVPDwithpriorDatasetV1(data.Dataset):
    """
    Video test dataset for CVPD with prior dataset recommended by ITU-T.
    For validation data: Disk IO is adopted.   
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()
        self.opts_dict = opts_dict
        # dataset paths
        self.gt_root = op.join(
            '/home/zhuqiang/STDF30/data/CVPD/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            '/home/zhuqiang/STDF30/data/CVPD/', 
            self.opts_dict['lq_path']
            )
        
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'mv_path': [],
            # 'pm_path': [],
            'rm_path': [],
            'pd_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*')))
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('_')[-1])
            lq_name_vid = name_vid
            lq_vid_path = op.join(self.lq_root,lq_name_vid)
            for iter_frm in range(nfs):
                lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)

    def __getitem__(self, index):
        # get gt frame (img)
        # print('img shape',op.join(self.data_info['gt_path'][index],'{:03d}.png'.format(self.data_info['gt_index'][index])))
        img = cv2.imread(op.join(self.data_info['gt_path'][index],'{:03d}.png'.format(self.data_info['gt_index'][index]+1)), cv2.IMREAD_UNCHANGED)
        # gt-path, gt-frame-index,  lq-path, lq-frame-index
        img_gt = np.expand_dims( np.squeeze(img), 2).astype(np.float32) / 255.  # (H W 1)

        # get resdidue frames (npys)
        res = np.load(op.join(self.data_info['lq_path'][index],'residual/{:03d}.npy'.format(self.data_info['gt_index'][index]+1)))
        img_residues = res.transpose(2,1,0)

        # get resdidue frames (npys)
        # img_residues = []
        # for pred_index in self.data_info['lq_indexes'][index]:
        #     res = np.load(op.join(self.data_info['lq_path'][index],'residual/{:03d}.npy'.format(self.data_info['gt_index'][index]+1)))
        #     res = res.transpose(2,1,0)
        #     img_residues.append(res)
        # get lq frames (imgs)
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            img = cv2.imread(op.join(self.data_info['lq_path'][index],'compress_y/{:03d}.png'.format(lq_index+1)), cv2.IMREAD_UNCHANGED)
            img_lq = np.expand_dims( np.squeeze(img), 2 ).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)

        # get pred frames (imgs)
        img_preds = []
        for pred_index in self.data_info['lq_indexes'][index]:
            img = cv2.imread(op.join(self.data_info['lq_path'][index],'pred_y/{:03d}.png'.format(lq_index+1)), cv2.IMREAD_UNCHANGED)
            img_pred = np.expand_dims( np.squeeze(img), 2).astype(np.float32) / 255.  # (H W 1)
            img_preds.append(img_pred)
        
        # get PAI frames (imgs)
        # img_pais = []
        # for pai_index in self.data_info['lq_indexes'][index]:
        #     img = cv2.imread(op.join(self.data_info['lq_path'][index],'PAI/{:03d}.png'.format(lq_index+1)), cv2.IMREAD_UNCHANGED)
        #     img_pai = np.expand_dims( np.squeeze(img), 2).astype(np.float32) / 255.  # (H W 1)
        #     img_pais.append(img_pai)

        # get MV frames (npys)
        img_mvs = []
        for pred_index in self.data_info['lq_indexes'][index]:
            mv = np.load(op.join(self.data_info['lq_path'][index],'mv/{:03d}.npy'.format(lq_index+1)))
            mv = mv.transpose(1,2,0)
            img_mvs.append(mv)

        
        
        # no any augmentation
        # to tensor   #  需要修改 
        # print('[type]',img_gt.shape,img_preds[0].shape,img_mvs[0].shape,img_residues[0].shape)
        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]
        img_residues = totensor(img_residues)
        img_preds = torch.stack(totensor(img_preds), dim=0)
        # img_pais = torch.stack(totensor(img_pais), dim=0)
        img_mvs = torch.stack(totensor(img_mvs), dim=0)
        # img_residues = torch.stack(totensor(img_residues), dim=0)
        
        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            # 'PAI': img_pais,  # (1 H W)
            'pred': img_preds,  # (1 H W)
            'mv': img_mvs,  # (2 H W)
            'residue': img_residues,  # (1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num



class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
            # self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    def unpadx2(self,x):
        ht, wd = x.shape[-2:]
        c = [2*self._pad[2], ht-2*self._pad[3], 2*self._pad[0], wd-2*self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    def unpadx3(self,x):
        ht, wd = x.shape[-2:]
        c = [3*self._pad[2], ht-3*self._pad[3], 3*self._pad[0], wd-3*self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

    def unpadx4(self,x):
        ht, wd = x.shape[-2:]
        c = [4*self._pad[2], ht-4*self._pad[3], 4*self._pad[0], wd-4*self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

