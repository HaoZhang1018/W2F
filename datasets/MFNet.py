from .base import CustomDataset, EnhanceDataset

def get_class_colors():
    pattale = [
        [0, 0, 0],  # unlabelled
        [128, 0, 64],  # car
        [0, 64, 64],  # person
        [192, 128, 0],  # bike
        [192, 0, 0],  # curve
        [0, 128, 128],  # car_stop
        [128, 64, 64],  # guardrail
        [128, 128, 192],  # color_cone
        [0, 64, 192],  # bump
    ]
    return pattale

class MFNetDataset(CustomDataset):
    def __init__(self, cfg, preprocess=None, stage="train", ignore_index=255, palette=None):
        super().__init__(
            data_root=cfg.datasets.root,
            height = cfg.datasets.image_height,
            width = cfg.datasets.image_width,
            path1=cfg.datasets.path1,
            path1_suffix=cfg.datasets.path1_suffix,
            path2=cfg.datasets.path2,
            path2_suffix=cfg.datasets.path2_suffix,
            label=cfg.datasets.label,
            label_suffix=cfg.datasets.label_suffix,
            stage=stage,
            split=cfg.datasets.split,
            ignore_index=ignore_index,
            preprocess=preprocess,
            classes=cfg.datasets.class_names,
            palette=palette
        )
        
        if palette is None:
            self.palette = get_class_colors()


class MFNetEnhanceDataset(EnhanceDataset):
    def __init__(self, cfg, preprocess=None, stage="train", ignore_index=255, palette=None):
        super().__init__(
            data_root=cfg.datasets.root,
            height = cfg.datasets.image_height,
            width = cfg.datasets.image_width,
            path1=cfg.datasets.path1,
            path1_HQ=cfg.datasets.path1_HQ,
            path1_suffix=cfg.datasets.path1_suffix,
            path2=cfg.datasets.path2,
            path2_HQ=cfg.datasets.path2_HQ,
            path2_suffix=cfg.datasets.path2_suffix,
            label=cfg.datasets.label,
            label_suffix=cfg.datasets.label_suffix,
            stage=stage,
            split=cfg.datasets.split,
            ignore_index=ignore_index,
            preprocess=preprocess,
            classes=cfg.datasets.class_names,
            palette=palette
        )
        
        if palette is None:
            self.palette = get_class_colors()