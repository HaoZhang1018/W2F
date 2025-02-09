import numpy as np

 
def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))
    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                        minlength=n_cl ** 2).reshape(n_cl, n_cl)
    
    return confusionMatrix, labeled, correct

def compute_score(hist, correct, labeled):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    iou = np.nan_to_num(iou)
    mean_IoU = np.nanmean(iou)

    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

    classAcc = np.diag(hist) / hist.sum(axis=1)
    classAcc = np.nan_to_num(classAcc)
    mean_pixel_acc = np.nanmean(classAcc)

    pixel_acc = correct / labeled

    return iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc, classAcc

    