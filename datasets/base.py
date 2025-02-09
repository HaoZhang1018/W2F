import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import cv2
import mmcv

from .utils import normalize
from .utils import random_mirror, random_scale, generate_random_crop_pos, random_crop_pad_to_shape
from .utils import ColorAugSSDTransform, MaskGenerator

class FusionDataset(Dataset):
    def __init__(
            self,
            data_root,
            path1="visible",
            path2="infrared",
            ):
        self.path1 = os.path.join(data_root, path1)
        self.path2 = os.path.join(data_root, path2)
        self.img_infos = self.load_annotations()

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)
    
    def load_annotations(self):
        img_infos = []
        for img in os.listdir(self.path1):
            img_name = img.split(".")[0]
            img_info = dict(filename=img_name)
            img_info['path1'] = os.path.join(self.path1, img)
            img_info['path2'] = os.path.join(self.path2, img)
            img_infos.append(img_info)
        
        return img_infos
    
    @staticmethod
    def _open_image(filepath, dtype=None):
        img = cv2.imread(filepath, -1)
        if len(img.shape) == 2:
            img = cv2.merge([img, img, img])
        elif len(img.shape) == 3:
            img = img[:,:,[2,1,0]]
        else:
            raise "Not Implement error"
        return np.array(img, dtype=dtype)
    
    @staticmethod
    def process_imgs(img):
        img = normalize(img)
        # reflect padding
        img = np.pad(img, pad_width=((32, 32), (32, 32), (0, 0)), mode='reflect')
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(img[None, :, :, :])).float()
        return img
    
    def __getitem__(self, idx):
        image_info = self.img_infos[idx]
        img1 = self._open_image(image_info["path1"], dtype=np.uint8)
        img2 = self._open_image(image_info["path2"], dtype=np.uint8)

        image_size = img1.shape[:-1]

        img1 = self.process_imgs(img1)
        img2 = self.process_imgs(img2)
        return img1, img2, image_size, image_info["filename"]


class CustomDataset(Dataset):
    def __init__(
            self,
            data_root,
            height,
            width,
            path1="visible",
            path1_suffix="png",
            path2="infrared",
            path2_suffix="png",
            label="labels",
            label_suffix="png",
            stage='train',
            split=None,
            ignore_index=None,
            preprocess=None,
            classes=None,
            palette=None
            ):
        self.stage = stage
        self.img_size = (height, width)
        self.mosaic_border = [-height // 4, -width // 4]

        self.path1 = os.path.join(data_root, stage, path1)
        self.path2 = os.path.join(data_root, stage, path2)
        self.label = os.path.join(data_root, stage, label)

        self.img_infos = self.load_annotations(path1_suffix, path2_suffix, label_suffix, split)
        self.indexs = [i for i in range(len(self.img_infos))]
        self.preprocess = preprocess

        self.ignore_index = ignore_index
        self.classes = classes
        self.palette = palette

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)
    
    def load_annotations(self, path1_suffix, path2_suffix, label_suffix, split=None):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name)
                    img_info['path1'] = os.path.join(self.path1, img_name+path1_suffix)
                    img_info['path2'] = os.path.join(self.path2, img_name+path2_suffix)
                    img_info['label'] = os.path.join(self.label, img_name+label_suffix)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(self.path1, path1_suffix, recursive=True):
                img_name = img.replace(path1_suffix, "")
                img_info = dict(filename=img_name)
                img_info['path1'] = os.path.join(self.path1, img_name+path1_suffix)
                img_info['path2'] = os.path.join(self.path2, img_name+path2_suffix)
                img_info['label'] = os.path.join(self.label, img_name+label_suffix)
                img_infos.append(img_info)

        return img_infos

    def stitch_imgs(self, i, img, img4, yc, xc, h, w, s_h, s_w, gt=False):
        if i == 0:  # top left
            h_c, w_c = yc, xc
            if (h_c/w_c)  > (s_h/s_w):
                img = cv2.resize(img, (int(h_c*w/h), h_c), interpolation=cv2.INTER_NEAREST if gt else cv2.INTER_AREA)
                img4[:yc, :xc] = img[:, :w_c]
            else:
                img = cv2.resize(img, (w_c, int(w_c*h/w)), interpolation=cv2.INTER_NEAREST if gt else cv2.INTER_AREA)
                img4[:yc, :xc] = img[:h_c, :]
        elif i == 1:  # top right
            h_c, w_c = yc, s_w - xc
            if (h_c/w_c)  > (s_h/s_w):
                img = cv2.resize(img, (int(h_c*w/h), h_c), interpolation=cv2.INTER_NEAREST if gt else cv2.INTER_AREA)
                img4[:yc, xc:] = img[:, :w_c]
            else:
                img = cv2.resize(img, (w_c, int(w_c*h/w)), interpolation=cv2.INTER_NEAREST if gt else cv2.INTER_AREA)
                img4[:yc, xc:] = img[:h_c, :]
        elif i == 2:  # bottom left
            h_c, w_c = s_h - yc, xc
            if (h_c/w_c)  > (s_h/s_w):
                img = cv2.resize(img, (int(h_c*w/h), h_c), interpolation=cv2.INTER_NEAREST if gt else cv2.INTER_AREA)
                img4[yc:, :xc] = img[:, :w_c]
            else:
                img = cv2.resize(img, (w_c, int(w_c*h/w)), interpolation=cv2.INTER_NEAREST if gt else cv2.INTER_AREA)
                img4[yc:, :xc] = img[:h_c, :]
        elif i == 3:  # bottom left
            h_c, w_c = s_h - yc, s_w - xc
            if (h_c/w_c)  > (s_h/s_w):
                img = cv2.resize(img, (int(h_c*w/h), h_c), interpolation=cv2.INTER_NEAREST if gt else cv2.INTER_AREA)
                img4[yc:, xc:] = img[:, :w_c]
            else:
                img = cv2.resize(img, (w_c, int(w_c*h/w)), interpolation=cv2.INTER_NEAREST if gt else cv2.INTER_AREA)
                img4[yc:, xc:] = img[:h_c, :]
        return img4
    
    def load_mosaic(self, indices):
        s_h, s_w = self.img_size

        yc, xc = [int(random.uniform(-x, s + x)) for s, x in zip(self.img_size, self.mosaic_border)] # mosaic center x, y
        
        img1_4 = np.full((s_h, s_w, 3), 0, dtype=np.uint8)  # base image with 4 tiles
        img2_4 = np.full((s_h, s_w, 3), 0, dtype=np.uint8)  # base image with 4 tiles
        gt_4 = np.full((s_h, s_w), 0, dtype=np.uint8)  # base image with 4 tiles

        random.shuffle(indices)
        for i, idx in enumerate(indices):
            image_info = self.img_infos[idx]
            img1 = self._open_image(image_info["path1"], dtype=np.uint8)
            img2 = self._open_image(image_info["path2"], dtype=np.uint8)
            gt   = self._open_label(image_info["label"], dtype=np.uint8)

            h, w = img1.shape[:2]

            img1_4 = self.stitch_imgs(i, img1, img1_4, yc, xc, h, w, s_h, s_w)
            img2_4 = self.stitch_imgs(i, img2, img2_4, yc, xc, h, w, s_h, s_w)
            gt_4   = self.stitch_imgs(i, gt, gt_4, yc, xc, h, w, s_h, s_w, True)

        return img1_4, img2_4, gt_4

    @staticmethod
    def _open_image(filepath, dtype=None):
        img = cv2.imread(filepath, -1)
        if len(img.shape) == 2:
            img = cv2.merge([img, img, img])
        elif len(img.shape) == 3:
            img = img[:,:,[2,1,0]]
        else:
            raise "Not Implement error"
        return np.array(img, dtype=dtype)
    
    @staticmethod
    def _open_label(filepath, dtype=np.uint8):
        img = cv2.imread(filepath, -1)
        return np.array(img, dtype=dtype)

    def __getitem__(self, idx):
        if self.preprocess is not None and self.stage == 'train':
            if random.random() > 0.5:
                indexs = [idx] + random.choices(self.indexs, k=3)
                img1, img2, gt = self.load_mosaic(indexs)
            else:
                image_info = self.img_infos[idx]
                img1 = self._open_image(image_info["path1"], dtype=np.uint8)
                img2 = self._open_image(image_info["path2"], dtype=np.uint8)
                gt = self._open_label(image_info["label"], dtype=np.uint8)

            img1, img2, gt, Mask = self.preprocess(img1, img2, gt)
            
            img1 = torch.from_numpy(np.ascontiguousarray(img1)).float()
            img2 = torch.from_numpy(np.ascontiguousarray(img2)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            
            output_dict = dict(modal_x=img1, modal_y=img2, label=gt, Mask=Mask)

        else:
            image_info = self.img_infos[idx]
            img1 = self._open_image(image_info["path1"], dtype=np.uint8)
            img2 = self._open_image(image_info["path2"], dtype=np.uint8)
            gt = self._open_label(image_info["label"], dtype=np.uint8)

            output_dict = dict(modal_x=img1, modal_y=img2, label=gt, fn=image_info["filename"])
        
        return output_dict
        

class EnhanceDataset(CustomDataset):
    def __init__(
            self,
            data_root,
            height,
            width,
            path1="visible",
            path1_HQ="visible_HQ",
            path1_suffix="png",
            path2="infrared",
            path2_HQ="infrared_HQ",
            path2_suffix="png",
            label="labels",
            label_suffix="png",
            stage='train',
            split=None,
            ignore_index=None,
            preprocess=None,
            classes=None,
            palette=None
            ):
        self.stage = stage
        self.img_size = (height, width)
        self.mosaic_border = [-height // 4, -width // 4]
        
        self.path1 = os.path.join(data_root, stage, path1)
        self.path2 = os.path.join(data_root, stage, path2)
        self.label = os.path.join(data_root, stage, label)
        self.path1_HQ = os.path.join(data_root, stage, path1_HQ)
        self.path2_HQ = os.path.join(data_root, stage, path2_HQ)
        
        self.img_infos = self.load_annotations(path1_suffix, path2_suffix, label_suffix, split)
        self.indexs = [i for i in range(len(self.img_infos))]
        self.preprocess = preprocess

        self.ignore_index = ignore_index
        self.classes = classes
        self.palette = palette

    def load_annotations(self, path1_suffix, path2_suffix, label_suffix, split=None):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name)
                    img_info['path1_LQ'] = os.path.join(self.path1, img_name+path1_suffix)
                    img_info['path2_LQ'] = os.path.join(self.path2, img_name+path2_suffix)
                    img_info['path1_HQ'] = os.path.join(self.path1_HQ, img_name+path1_suffix)
                    img_info['path2_HQ'] = os.path.join(self.path2_HQ, img_name+path2_suffix)
                    img_info['label'] = os.path.join(self.label, img_name+label_suffix)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(self.path1, path1_suffix, recursive=True):
                img_name = img.replace(path1_suffix, "")
                img_info = dict(filename=img_name)
                img_info['path1_LQ'] = os.path.join(self.path1, img_name+path1_suffix)
                img_info['path2_LQ'] = os.path.join(self.path2, img_name+path2_suffix)
                img_info['path1_HQ'] = os.path.join(self.path1_HQ, img_name+path1_suffix)
                img_info['path2_HQ'] = os.path.join(self.path2_HQ, img_name+path2_suffix)
                img_info['label'] = os.path.join(self.label, img_name+label_suffix)
                img_infos.append(img_info)

        return img_infos
    
    def load_mosaic(self, indices):
        s_h, s_w = self.img_size

        yc, xc = [int(random.uniform(-x, s + x)) for s, x in zip(self.img_size, self.mosaic_border)] # mosaic center x, y
        
        img1_4 = np.full((s_h, s_w, 3), 0, dtype=np.uint8)  # base image with 4 tiles
        img2_4 = np.full((s_h, s_w, 3), 0, dtype=np.uint8)  # base image with 4 tiles
        img1_HQ_4 = np.full((s_h, s_w, 3), 0, dtype=np.uint8)  # base image with 4 tiles
        img2_HQ_4 = np.full((s_h, s_w, 3), 0, dtype=np.uint8)  # base image with 4 tiles
        gt_4 = np.full((s_h, s_w), 0, dtype=np.uint8)  # base image with 4 tiles

        random.shuffle(indices)
        for i, idx in enumerate(indices):
            image_info = self.img_infos[idx]
            img1 = self._open_image(image_info["path1_LQ"], dtype=np.uint8)
            img2 = self._open_image(image_info["path2_LQ"], dtype=np.uint8)
            img1_HQ = self._open_image(image_info["path1_HQ"], dtype=np.uint8)
            img2_HQ = self._open_image(image_info["path2_HQ"], dtype=np.uint8)
            gt   = self._open_label(image_info["label"], dtype=np.uint8)

            h, w = img1.shape[:2]

            img1_4 = self.stitch_imgs(i, img1, img1_4, yc, xc, h, w, s_h, s_w)
            img2_4 = self.stitch_imgs(i, img2, img2_4, yc, xc, h, w, s_h, s_w)
            img1_HQ_4 = self.stitch_imgs(i, img1_HQ, img1_HQ_4, yc, xc, h, w, s_h, s_w)
            img2_HQ_4 = self.stitch_imgs(i, img2_HQ, img2_HQ_4, yc, xc, h, w, s_h, s_w)
            gt_4   = self.stitch_imgs(i, gt, gt_4, yc, xc, h, w, s_h, s_w, True)

        return img1_4, img2_4, img1_HQ_4, img2_HQ_4, gt_4

    def __getitem__(self, idx):
        if self.preprocess is not None and self.stage == 'train':
            if random.random() > 0.5:
                indexs = [idx] + random.choices(self.indexs, k=3)
                img1, img2, img1_HQ, img2_HQ, gt = self.load_mosaic(indexs)
            else:
                image_info = self.img_infos[idx]
                img1 = self._open_image(image_info["path1_LQ"], dtype=np.uint8)
                img2 = self._open_image(image_info["path2_LQ"], dtype=np.uint8)
                img1_HQ = self._open_image(image_info["path1_HQ"], dtype=np.uint8)
                img2_HQ = self._open_image(image_info["path2_HQ"], dtype=np.uint8)
                gt = self._open_label(image_info["label"], dtype=np.uint8)   
            
            combine_img1 = np.concatenate((img1, img1_HQ), axis=-1)
            combine_img2 = np.concatenate((img2, img2_HQ), axis=-1)

            combine_img1, combine_img2, gt, Mask = self.preprocess(combine_img1, combine_img2, gt)
            img1 = combine_img1[0:3]
            img1_HQ = combine_img1[3:]
            img2 = combine_img2[0:3]
            img2_HQ = combine_img2[3:]

            img1 = torch.from_numpy(np.ascontiguousarray(img1)).float()
            img2 = torch.from_numpy(np.ascontiguousarray(img2)).float()
            img1_HQ = torch.from_numpy(np.ascontiguousarray(img1_HQ)).float()
            img2_HQ = torch.from_numpy(np.ascontiguousarray(img2_HQ)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            
            output_dict = dict(modal_x=img1, modal_y=img2, label_x=img1_HQ, label_y=img2_HQ, label=gt, Mask=Mask)
        
        return output_dict


class Train_pipline(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_scale_array = cfg.train.train_scale_array
        self.image_height = cfg.datasets.image_height
        self.image_width = cfg.datasets.image_width

        self.color_augment = ColorAugSSDTransform(img_format="RGB")
        self.mask_generator = MaskGenerator(input_size=[self.image_height, self.image_width],
                                            mask_patch_size=cfg.train.Mask.size,
                                            strategy=cfg.train.Mask.strategy
                                            )
    
    def __call__(self, img1, img2, gt):
        if img1.shape[-1] > 3:
            img1[...,:3] = self.color_augment.apply_image(img1[...,:3])
            
            img1, img2, gt = random_mirror(img1, img2, gt)
            if self.train_scale_array is not None:
                img1, img2, gt, scale = random_scale(img1, img2, gt, self.train_scale_array)

            img1 = normalize(img1)
            img2 = normalize(img2)

            crop_size = (self.image_height, self.image_width)
            crop_pos = generate_random_crop_pos(img1.shape[:2], crop_size)

            p_img1, Margin = random_crop_pad_to_shape(img1, crop_pos, crop_size, 0)
            p_img2, _ = random_crop_pad_to_shape(img2, crop_pos, crop_size, 0)
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

            p_img1 = p_img1.transpose(2, 0, 1)
            p_img2 = p_img2.transpose(2, 0, 1)

            p_mask1, p_mask2 = self.mask_generator()
            
            p_img1[:3] = p_img1[:3] * p_mask1
            p_img2[:3] = p_img2[:3] * p_mask2
        
        else:
            img1 = self.color_augment.apply_image(img1)

            img1, img2, gt = random_mirror(img1, img2, gt)
            if self.train_scale_array is not None:
                img1, img2, gt, scale = random_scale(img1, img2, gt, self.train_scale_array)

            img1 = normalize(img1)
            img2 = normalize(img2)

            crop_size = (self.image_height, self.image_width)
            crop_pos = generate_random_crop_pos(img1.shape[:2], crop_size)

            p_img1, Margin = random_crop_pad_to_shape(img1, crop_pos, crop_size, 0)
            p_img2, _ = random_crop_pad_to_shape(img2, crop_pos, crop_size, 0)
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

            p_img1 = p_img1.transpose(2, 0, 1)
            p_img2 = p_img2.transpose(2, 0, 1)

            p_mask1, p_mask2 = self.mask_generator()

            p_img1 = p_img1 * p_mask1
            p_img2 = p_img2 * p_mask2

        Mask = np.zeros((3, p_img1.shape[1], p_img1.shape[2]))
        Mask[:, Margin[0]:(crop_size[0]-Margin[1]), Margin[2]:(crop_size[1]-Margin[3])] = 1.

        return p_img1, p_img2, p_gt, Mask.astype(np.float32)
    

class Train_search_pipline(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_scale_array = cfg.train.train_scale_array
        self.image_height = cfg.datasets.image_height
        self.image_width = cfg.datasets.image_width
    
    def __call__(self, img1, img2, gt):
        img1, img2, gt = random_mirror(img1, img2, gt)
        if self.train_scale_array is not None:
            img1, img2, gt, scale = random_scale(img1, img2, gt, self.train_scale_array)

        img1 = normalize(img1)
        img2 = normalize(img2)

        crop_size = (self.image_height, self.image_width)
        crop_pos = generate_random_crop_pos(img1.shape[:2], crop_size)

        p_img1, Margin = random_crop_pad_to_shape(img1, crop_pos, crop_size, 0)
        p_img2, _ = random_crop_pad_to_shape(img2, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

        p_img1 = p_img1.transpose(2, 0, 1)
        p_img2 = p_img2.transpose(2, 0, 1)
        
        Mask = np.zeros((3, p_img1.shape[1], p_img1.shape[2]))
        Mask[:, Margin[0]:(crop_size[0]-Margin[1]), Margin[2]:(crop_size[1]-Margin[3])] = 1.

        return p_img1, p_img2, p_gt, Mask.astype(np.float32)