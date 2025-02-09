from .base import CustomDataset, EnhanceDataset

def get_class_colors():
    pattale = [
        [0, 0, 0],  # background  0
        [228, 228, 179],  # Road 1
        [133, 57, 181],  # Sidewalk 2
        [177, 162, 67],  # Building 3
        [50, 178, 200],  # Lamp 4
        [199, 45, 132],  # Sign 5
        [84, 172, 66],  # Vegetation 6
        [79, 73, 179],  # Sky 7
        [166, 99, 76],  # Person 8
        [253, 121, 66],  # Car 9
        [91, 165, 137],  # Truck 10
        [152, 97, 155],  # Bus 11
        [140, 153, 105],  # Motorcycle 12
        [158, 215, 222],  # Bicycle 13
        [90, 113, 135],  # Pole 14
    ]
    return pattale

class FMBDataset(CustomDataset):
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


class FMBEnhanceDataset(EnhanceDataset):
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