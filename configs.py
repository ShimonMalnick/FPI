import numbers
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, Sequence, Union, List
from torchvision.transforms.v2 import ScaleJitter, RandomCrop, RandomAffine, RandomPerspective, \
    RandomVerticalFlip, RandomChannelPermutation, RandomPhotometricDistort, GaussianBlur


@dataclass
class BaseDatasetConfig:
    batch_size: int = 32
    num_ddim_steps: int = 10
    middle_latent_step: int = 4
    num_iter_fixed_point: int = 5
    save_dir: Optional[str] = None
    dataset_indices: Optional[Tuple[int, int]] = None
    GUIDANCE_SCALE: float = 2.0
    name: Optional[str] = None
    transform: Optional[Callable] = None

    def to_dict(self):
        d = self.__dict__.copy()
        # popping un hashable values
        d.pop('transform')
        return d


@dataclass
class CocoDatasetConfig(BaseDatasetConfig):
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_coco")
    dataset_indices = (0, 500)


@dataclass
class ChestXRayDatasetConfig(BaseDatasetConfig):
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_chest_xray")
    dataset_indices = (0, 500)


@dataclass
class ImageNetSubsetDatasetConfig(BaseDatasetConfig):
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_imagenet_subset")
    split: str = 'train'
    num_classes: int = 100
    num_images_per_class: int = 100


@dataclass
class NormalDistributedDatasetDatasetConfig(BaseDatasetConfig):
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_normal_distributed_dataset")
    ds_size: int = 500
    images_shape: Tuple[int] = (3, 512, 512)


@dataclass
class NoisedCocoCaptions17DatasetConfig(BaseDatasetConfig):
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_coco_noised")
    num_images_before_noise: int = 100
    num_noise_levels: int = 30
    noise_multiplier: float = 6.0


@dataclass
class AugmentationConfig:
    augmentation_func: Callable


@dataclass
class ScaleJitterConfig(AugmentationConfig):
    target_size: Tuple[int, int] = (512, 512)
    scale_range: Tuple[float, float] = (0.5, 2.0)
    augmentation_func: Callable = ScaleJitter

    def __str__(self):
        return f"ScaleJitter(target_size={self.target_size}, scale_range={self.scale_range})"


@dataclass
class RandomCropConfig(AugmentationConfig):
    size: Union[int, Sequence[int]] = (256, 256)
    augmentation_func: Callable = RandomCrop

    def __str__(self):
        return f"RandomCrop(size={self.size})"


@dataclass
class RandomAffineConfig(AugmentationConfig):
    degrees: Union[numbers.Number, Sequence] = 90
    augmentation_func: Callable = RandomAffine

    def __str__(self):
        return f"RandomAffine(degrees={self.degrees})"


@dataclass
class RandomPerspectiveConfig(AugmentationConfig):
    augmentation_func: Callable = RandomPerspective

    def __str__(self):
        return "RandomPerspective"


@dataclass
class RandomVerticalFlipConfig(AugmentationConfig):
    p: float = 1.0  # used not randomly, i.e. flipping the image vertically
    augmentation_func: Callable = RandomVerticalFlip

    def __str__(self):
        return "RandomVerticalFlip"


@dataclass
class RandomChannelPermutationConfig(AugmentationConfig):
    augmentation_func: Callable = RandomChannelPermutation

    def __str__(self):
        return "RandomChannelPermutation"


@dataclass
class RandomPhotometricDistortConfig(AugmentationConfig):
    augmentation_func: Callable = RandomPhotometricDistort

    def __str__(self):
        return "RandomPhotometricDistort"


@dataclass
class GaussianBlurConfig(AugmentationConfig):
    kernel_size: Union[int, Sequence[int]] = 25
    augmentation_func: Callable = GaussianBlur

    def __str__(self):
        return f"GaussianBlur(kernel_size={self.kernel_size})"


@dataclass
class IdentityConfig(AugmentationConfig):
    augmentation_func: Callable = lambda **kwargs: lambda x: x

    def __str__(self):
        return "Identity"


@dataclass
class AugmentedDatasetConfig(BaseDatasetConfig):
    dataset_config: BaseDatasetConfig = None
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_augmented")
    augmentations: List[AugmentationConfig] = field(
        default_factory=lambda: [ScaleJitterConfig(), RandomCropConfig(), RandomAffineConfig(),
                                 RandomPerspectiveConfig(),
                                 RandomVerticalFlipConfig(), RandomChannelPermutationConfig(),
                                 RandomPhotometricDistortConfig(),
                                 GaussianBlurConfig()])
    augmentation_config: Optional[Callable] = None

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def to_dict(self):
        d = super().to_dict()
        d['augmentations'] = [str(aug) for aug in self.augmentations]
        d['dataset_config'] = self.dataset_config.to_dict()
        d['dataset_config']['name'] = type(self.dataset_config).__name__
        return d
