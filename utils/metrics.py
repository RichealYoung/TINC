import os
import cv2
import numpy as np
from utils.tool import get_type_max, read_img, save_img
from omegaconf import OmegaConf
import torch
import json
import sys
from tqdm import tqdm
from einops import rearrange, repeat
from utils.ssim import ssim as ssim_calc
from utils.ssim import ms_ssim as ms_ssim_calc
import copy

def cal_iou_acc_pre(data_gt:np.ndarray,data_hat:np.ndarray,thres:float=1):
    hat = np.copy(data_hat)
    gt = np.copy(data_gt)
    hat[data_hat>=thres]=1
    hat[data_hat<thres]=0
    gt[data_gt>=thres]=1
    gt[data_gt<thres]=0
    tp = (gt*hat).sum()
    tn = ((gt+hat)==0).sum()
    fp = ((gt==0)*(hat==1)).sum()
    fn = ((gt==1)*(hat==0)).sum()
    iou = 1.0*tp/(tp+fp+fn)
    acc = 1.0*(tp+tn)/(tp+fp+tn+fn)
    pre = 1.0*tp/(tp+fp)
    return iou, acc, pre

def cal_psnr(data_gt:np.ndarray, data_hat:np.ndarray, data_range):
    data_gt = np.copy(data_gt)
    data_hat = np.copy(data_hat)
    mse = np.mean(np.power(data_gt/data_range-data_hat/data_range,2))
    psnr = -10*np.log10(mse)
    return psnr

def eval_performance(orig_data, decompressed_data):
    max_range = get_type_max(orig_data)
    orig_data = orig_data.astype(np.float32)
    decompressed_data = decompressed_data.astype(np.float32)
    # accuracy
    acc200 = cal_iou_acc_pre(orig_data, decompressed_data, thres=200)[1]
    acc500 = cal_iou_acc_pre(orig_data, decompressed_data, thres=500)[1]
    # psnr
    psnr_value = cal_psnr(orig_data, decompressed_data, max_range)
    # ssim
    orig_data = torch.from_numpy(orig_data)
    decompressed_data = torch.from_numpy(decompressed_data)
    # convert to NCHW or NCDHW
    if len(orig_data.shape) == 3:
        data1 = rearrange(orig_data, 'h w (n c) -> n c h w', n=1)
        data2 = rearrange(decompressed_data, 'h w (n c) -> n c h w', n=1)
        ssim_value = ssim_calc(data1, data2, max_range)
    elif len(orig_data.shape) == 4:
        ssim_value_total = 0 
        for i in tqdm(range(orig_data.shape[0]), desc='Evaluating', leave=False, file=sys.stdout):
            data1 = copy.deepcopy(orig_data[i])
            data2 = copy.deepcopy(decompressed_data[i])
            data1 = rearrange(data1, 'h w (n c) -> n c h w', n=1)
            data2 = rearrange(data2, 'h w (n c) -> n c h w', n=1)
            ssim_value_total += ssim_calc(data1, data2, max_range)
        ssim_value = ssim_value_total/orig_data.shape[0]
    else:
        raise ValueError
    
    return psnr_value, float(ssim_value), acc200, acc500