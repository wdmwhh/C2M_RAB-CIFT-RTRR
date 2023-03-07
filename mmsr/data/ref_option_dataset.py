import cv2
import os
import numpy as np
import glob
import random
import mmcv
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmsr.data.transforms import mod_crop, totensor


class OptionSet(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_list = sorted(os.listdir(opt['dataroot']))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        scale = self.opt['scale']

        filename = self.input_list[idx]
        dirname = os.path.join(self.opt['dataroot'], filename)
        img_in = cv2.imread(os.path.join(dirname, filename))
        img_ref = cv2.imread(glob.glob(os.path.join(dirname, "00*"))[0])

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()
        img_ref = mod_crop(img_ref, scale)
        img_in_h, img_in_w, _ = img_in.shape
        img_ref_h, img_ref_w, _ = img_ref.shape
        padding = False

        if img_in_h != img_ref_h or img_in_w != img_ref_w:
            padding = True
            target_h = max(img_in_h, img_ref_h)
            target_w = max(img_in_w, img_ref_w)
            img_in = mmcv.impad(
                img_in, shape=(target_h, target_w), pad_val=0)
            img_ref = mmcv.impad(
                img_ref, shape=(target_h, target_w), pad_val=0)

        gt_h, gt_w, _ = img_in.shape
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = Image.fromarray(img_ref).resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        img_in = img_in.astype(np.float32) / 255.
        img_ref = img_ref.astype(np.float32) / 255.
        img_in_gt = img_in_gt.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        img_ref_lq = np.array(img_ref_lq).astype(np.float32) / 255.
        img_ref_up = np.array(img_ref_up).astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt = totensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_ref, img_ref_lq, img_ref_up, img_in_gt],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_in': img_in_gt,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_ref_lq': img_ref_lq,
            'img_ref_up': img_ref_up,
            'lq_path': dirname,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict

