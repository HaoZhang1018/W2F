import numpy as np
import torch.nn.functional as F

from PIL import Image


label_colours = [(178, 45, 45), (153, 115, 115), (64, 36, 32), (255, 68, 0), (89, 24, 0), (191, 121, 96), (191, 102, 0),
                 (76, 41, 0), (153, 115, 38), (102, 94, 77), (242, 194, 0), (191, 188, 143), (226, 242, 0),
                 (119, 128, 0), (59, 64, 0), (105, 191, 48), (81, 128, 64), (0, 255, 0), (0, 51, 7), (191, 255, 208),
                 (96, 128, 113), (0, 204, 136), (13, 51, 43), (0, 191, 179), (0, 204, 255), (29, 98, 115), (0, 34, 51),
                 (163, 199, 217), (0, 136, 255), (41, 108, 166), (32, 57, 128), (0, 22, 166), (77, 80, 102),
                 (119, 54, 217), (41, 0, 77), (222, 182, 242), (103, 57, 115), (247, 128, 255), (191, 0, 153),
                 (128, 96, 117), (127, 0, 68), (229, 0, 92), (76, 0, 31), (255, 128, 179), (242, 182, 198)]


def normalize(x):
    return x / np.linalg.norm(x, ord=2, axis=0, keepdims=True)

 
def process_image(image, image_mean):
    image = image.cpu().numpy() + image_mean[:, None, None]
    return image.astype(np.uint8)


def process_seg_label(pred, gt, num_classes=40):
    B, C, H, W = gt.size()
    pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)
    pred = pred.argmax(dim=1)[0].detach()
    gt = gt.squeeze(1)[0]
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    h, w = gt.shape
    pred_img = Image.new('RGB', (w, h), (255, 255, 255))  # unlabeled part is white (255, 255, 255)
    gt_img = Image.new('RGB', (w, h), (255, 255, 255))
    pred_pixels = pred_img.load()
    gt_pixels = gt_img.load()
    for j_, j in enumerate(gt):
        for k_, k in enumerate(j):
            if k < num_classes:
                gt_pixels[k_, j_] = label_colours[k]
                pred_pixels[k_, j_] = label_colours[pred[j_, k_]]
    return np.array(pred_img).transpose([2, 0, 1]), np.array(gt_img).transpose([2, 0, 1])


def process_fusion_imgs(img1, img2, fus):
    img1 = img1[0].cpu().numpy()
    img2 = img2[0].cpu().numpy()
    fus  = fus[0].cpu().numpy()

    img1 = np.array(img1).transpose([2, 0, 1])
    img2 = np.array(img2).transpose([2, 0, 1])
    fus = np.array(fus).transpose([2, 0, 1])
    return img1, img2, fus


def print_iou(metrics, class_names=None, show_no_back=False, print=False):
    iou = metrics["Class iou"]
    mean_pixel_acc = metrics["Mean Acc"]
    pixel_acc = metrics["Pixel Acc"]
    class_acc = metrics["Class Acc"]

    n = iou.size
    lines = ['']
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.3f%%\t%.3f%%' % (cls, iou[i] * 100, class_acc[i] * 100))
    mean_IoU = np.nanmean(np.nan_to_num(iou))
    if show_no_back:
        mean_IoU_no_back = np.nanmean(iou[1:].nan_to_num())
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'mean_IU_no_back', mean_IoU_no_back*100,
                                                                                                'mean_pixel_acc', mean_pixel_acc*100, 'pixel_acc',pixel_acc*100))
    else:
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100,
                                                                                    'mean_pixel_acc', mean_pixel_acc*100, 'pixel_acc',pixel_acc*100))
    line = "\n".join(lines)
    if print:
        print(line)
    return line




