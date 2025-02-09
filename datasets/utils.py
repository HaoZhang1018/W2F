import cv2
import numpy as np

import random
import collections


def normalize(img):
    # pytorch pretrained model need the input range: 0-1
    img = img / 255.0
    return img.astype(np.float32)


def random_mirror(img1, img2, gt):
    if random.random() >= 0.5:
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
        gt = cv2.flip(gt, 1)

    return img1, img2, gt


def random_scale(img1, img2, gt, scales):
    scale = random.choice(scales)
    sh = int(img1.shape[0] * scale)
    sw = int(img1.shape[1] * scale)
    img1 = cv2.resize(img1, (sw, sh), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    
    return img1, img2, gt, scale


def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, collections.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape


def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)

    return img, margin


def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w


def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h:start_crop_h + crop_h,
               start_crop_w:start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)

    return img_, margin


class MaskGenerator:
    def __init__(self, input_size=[256, 320], mask_patch_size=32, \
                 mask_ratio=0.6, mask_type='patch', strategy='comp'):
        self.input_size = np.array(input_size)
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size[0] % self.mask_patch_size == 0
        assert self.input_size[1] % self.mask_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        
        self.token_count = self.rand_size[0] * self.rand_size[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        if mask_type == 'patch':
            self.gen_mask = self.gen_patch_mask
        else:
            raise AssertionError("Not valid mask type!")

        if strategy == 'none':
            self.strategy = self.gen_nomasks
        elif strategy == 'comp':
            self.strategy = self.gen_comp_masks
        elif strategy == 'rand_comp':
            self.strategy = self.gen_rand_comp_masks
        else:
            raise AssertionError("Not valid strategy!")
 
    def gen_patch_mask(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size[0], self.rand_size[1]))
        mask = np.expand_dims(mask.repeat(self.mask_patch_size, axis=0).repeat(self.mask_patch_size, axis=1), axis=0)
        mask = np.concatenate((mask, mask, mask), axis=0) # 3, ...

        return mask

    def gen_nomasks(self):
        return 1, 1
    
    def gen_comp_masks(self):
        mask = self.gen_mask()
        return mask, 1-mask

    def gen_rand_comp_masks(self):
        mask = self.gen_mask()
        nomask = np.ones_like(mask)

        idx = random.randrange(3)
        if idx == 0:   return nomask, mask
        elif idx == 1: return mask, nomask
        elif idx == 2: return mask, 1-mask

    def __call__(self):
        return self.strategy()

# Modified from Mask2former ColorAugSSDTransform function to take 4 channel input
class ColorAugSSDTransform:
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__()
        assert img_format in ["BGR", "RGB", "RGBT"]
        self.is_rgb = img_format == "RGB"
        self.is_rgbt = img_format == "RGBT"
        del img_format

        self.brightness_delta = brightness_delta
        self.contrast_high = contrast_high
        self.contrast_low = contrast_low
        self.saturation_high = saturation_high
        self.saturation_low = saturation_low
        self.hue_delta = hue_delta

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgbt:
            img_r = img[:, :, [2, 1, 0]]
            img_t = img[:, :, -1]
            img_r = self.brightness(img_r)
            if random.randrange(2):
                img_r = self.contrast(img_r)
                img_r = self.saturation(img_r)
                img_r = self.hue(img_r)
                img_t = self.contrast(img_t)
            else:
                img_r = self.saturation(img_r)
                img_r = self.hue(img_r)
                img_r = self.contrast(img_r)
                img_t = self.contrast(img_t)
            img[:,:,:3]  = img_r[:, :, [2, 1, 0]]
            img[:,:,-1] = img_t
        else:
            if self.is_rgb:
                img = img[:, :, [2, 1, 0]]
            img = self.brightness(img)
            if random.randrange(2):
                img = self.contrast(img)
                img = self.saturation(img)
                img = self.hue(img)
            else:
                img = self.saturation(img)
                img = self.hue(img)
                img = self.contrast(img)
            if self.is_rgb:
                img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randrange(2):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img):
        if random.randrange(2):
            return self.convert(img, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return img

    def saturation(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(self.saturation_low, self.saturation_high)
            )
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img


if __name__ == "__main__":
    mpdel = MaskGenerator([480, 640], 32, 0.6, "patch", "rand_comp")
    mpdel()
