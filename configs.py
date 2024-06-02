import os
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BaseConfig:
    batch_size: int = 16
    num_ddim_steps: int = 10
    middle_latent_step: int = 4
    num_iter_fixed_point: int = 5
    save_dir: Optional[str] = None
    dataset_indices: Optional[Tuple[int]] = None


@dataclass
class CocoConfig(BaseConfig):
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_coco")


@dataclass
class ChestXRayConfig(BaseConfig):
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_chest_xray")


@dataclass
class ImageNetSubsetConfig(BaseConfig):
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_imagenet_subset")
    split: str = 'train'
    num_classes: int = 100
    num_images_per_class: int = 100


@dataclass
class NormalDistributedDatasetConfig(BaseConfig):
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_normal_distributed_dataset")
    ds_size: int = 500
    images_shape: Tuple[int] = (3, 512, 512)


