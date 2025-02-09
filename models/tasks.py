import numpy as np

# 在这个文件的父目录
from utils.losses import seg_loss, fusion_loss
from utils.metrics import hist_info, compute_score
from utils.visualization import process_seg_label, process_fusion_imgs


class SegTask:
    def __init__(self, num_classes, weight, ignore_index):
        self.num_classes = num_classes
        self.seg_loss = seg_loss(weight, num_classes, ignore_index)

    def loss(self, prediction, gt):
        return self.seg_loss(prediction, gt)
    
    def log_visualize(self, prediction, gt, writer, steps):
        seg_pred, seg_gt = process_seg_label(prediction, gt, self.num_classes)
        writer.add_image('seg/pred', seg_pred, steps)
        writer.add_image('seg/gt', seg_gt, steps)

    def accumulate_metric(self, prediction, gt):
        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.num_classes, prediction, gt)
        accumulator = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        return accumulator
    
    def aggregate_metric(self, accumulator):
        hist = np.zeros((self.num_classes, self.num_classes))

        correct = 0
        labeled = 0
        count = 0
        for d in accumulator:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1
        
        iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc, classAcc = compute_score(hist, correct, labeled)
        return {
            'Class iou': iou,
            'Mean IoU': mean_IoU,
            'Pixel Acc': pixel_acc,
            'Class Acc': classAcc,
            'Mean Acc': mean_pixel_acc
        }
    

class FusionTask:
    def __init__(self, weight, method="MFNet", patch_size=64):
        self.fusion_loss = fusion_loss(weight, method, patch_size)

    def loss(self, fus, img1, img2, Mask, label):
        return self.fusion_loss(fus, img1, img2, Mask, label)
    
    def log_visualize(self, fus, img1, img2, writer, steps):
        img1, img2, fus = process_fusion_imgs(img1, img2, fus)
        writer.add_image('fusion/img1', img1, steps)
        writer.add_image('fusion/img2', img2, steps)
        writer.add_image('fusion/fus', fus, steps)