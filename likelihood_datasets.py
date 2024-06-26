import json
import os
from pathlib import Path
from typing import Union, Any, List, Tuple, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageNet
from setup import setup_config
from configs import AugmentationConfig, IdentityConfig

datasets_paths = {"COCO_ROOT": "mscoco17",
                  "CHEST_XRAY_ROOT": "chest_xray",
                  "IMAGENET_ROOT": "imagenet",
                  "VAN_GOGH": "vangogh_dataset"}

datasets_paths = {k: os.path.join(setup_config['DATASETS_ROOT'], v) for k, v in datasets_paths.items()}


def get_default_transform():
    return transforms.Compose(
        [lambda image: Image.open(image).convert('RGB').resize((512, 512)), transforms.ToTensor()])


class VanGoghDataset(Dataset):
    def __init__(self, tokenizer, in_distribution_dataset, root=datasets_paths["VAN_GOGH"], transform=None):
        self.images_root = root
        self.images_list = [os.path.join(root, p) for p in os.listdir(self.images_root)]
        self.dataset = in_distribution_dataset
        self.len = len(self.dataset)
        if transform is None:
            self.transform = get_default_transform()
        else:
            self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        example = {}
        image, caption = self.dataset[item % len(self.dataset)]
        example["instance_image"] = image
        if example["instance_image"].ndim == 4:
            assert example["instance_image"].shape[0] == 1
            example["instance_image"] = example["instance_image"][0]
        example['instance_van_gogh'] = self.transform(self.images_list[item % len(self.images_list)])
        if example["instance_van_gogh"].ndim == 4:
            assert example["instance_van_gogh"].shape[0] == 1
            example["instance_van_gogh"] = example["instance_van_gogh"][0]

        # the used prompt is always empty
        example["instance_prompt_ids"] = self.tokenizer(
            "",
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example


class ToyDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            static_data_root: str,
            in_distribution_dataset: Dataset,
            tokenizer,
            size: int = 512,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.dataset = in_distribution_dataset
        self.static_data_root = Path(static_data_root)
        if not self.static_data_root.exists():
            raise ValueError(f"Static data images root doesn't exists. got: {static_data_root}")

        self.static_images_path = list(Path(static_data_root).iterdir())
        self._length = len(self.dataset)
        self.static_data_pool = self.__create_static_data_pool()

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image, caption = self.dataset[index % len(self.dataset)]
        example["instance_image"] = image
        if example["instance_image"].ndim == 4:
            assert example["instance_image"].shape[0] == 1
            example["instance_image"] = example["instance_image"][0]
        example["static_intermediate_noise_preds"] = self.static_data_pool[index % len(self.static_images_path)]
        assert example["static_intermediate_noise_preds"].ndim == 4

        # the used prompt is always empty
        example["instance_prompt_ids"] = self.tokenizer(
            "",
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example

    def __create_static_data_pool(self):
        static_intermediate_noise_preds = [torch.load(self.static_images_path[i])["intermediate_noise_preds"] for i in
                                           range(len(self.static_images_path))]
        assert all([preds.ndim == 4 for preds in static_intermediate_noise_preds])
        return static_intermediate_noise_preds


class ImageNetSubset(ImageNet):
    def __init__(self, root: Union[str, Path] = datasets_paths['IMAGENET_ROOT'], split: str = 'train', num_classes=100,
                 num_images_per_class=100, **kwargs: Any):
        super().__init__(root, split, **kwargs)
        self._num_classes = num_classes
        self._num_images_per_class = num_images_per_class
        self.classes = self.classes[:num_classes]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}
        self.classes_indices = set(self.class_to_idx.values())
        self.samples = self.__create_samples_subset()
        self.targets = [s[1] for s in self.samples]
        if self.transform is None:
            self.transform = transforms.Compose(
                [lambda image: image.convert('RGB').resize((512, 512)), transforms.ToTensor()])

    def __create_samples_subset(self):
        samples = []
        # initialize counter for all the classes indices
        counters = {i: self._num_images_per_class for i in self.classes_indices}
        for name, label in self.samples:
            if label in self.classes_indices and counters[label] > 0:
                samples.append((name, label))
                counters[label] -= 1
        return samples


class CocoCaptions17(Dataset):
    def __init__(self, root=datasets_paths["COCO_ROOT"], transform=None):
        self.images_root = f"{root}/val2017"
        self.images_list = os.listdir(self.images_root)
        self.len = len(self.images_list)
        with open(f"{root}/annotations/captions_val2017.json") as f:
            self.captions_map = self._parse_captions(json.load(f)['annotations'])
        if transform is None:
            self.transform = get_default_transform()
        else:
            self.transform = transform

    @staticmethod
    def _parse_captions(captions):
        captions_map = {}
        for d in captions:
            img_id = d['image_id']
            if img_id not in captions_map:
                captions_map[img_id] = d['caption']
        return captions_map

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self._get_single_item(item)

    def _get_single_item(self, item):
        image_path = self.get_image_path_by_index(item)
        image = self.transform(image_path)
        image_id = int(self.images_list[item].split(".")[0])
        caption = self.captions_map[image_id]
        return image, caption

    def get_image_path_by_index(self, index):
        return os.path.join(self.images_root, self.images_list[index])


class ChestXRay(Dataset):
    def __init__(self, root=datasets_paths["CHEST_XRAY_ROOT"], transform=None):
        self.image_dirs = ["images_0" + ("0" + str(i) if i < 10 else str(i)) for i in range(1, 12)]
        self.data_root = root
        self.images = self.__create_images_data()
        self.static_caption = "A chest X-ray image"
        if transform is None:
            self.transform = get_default_transform()
        else:
            self.transform = transform

    def __create_images_data(self):
        images_list = []
        for image_dir in self.image_dirs:
            cur_images_root = os.path.join(self.data_root, image_dir, "images")
            cur_images = sorted(os.listdir(cur_images_root))
            images_list.extend([os.path.join(cur_images_root, im) for im in cur_images])
        return images_list

    def __getitem__(self, item):
        return self.transform(self.images[item]), self.static_caption

    def __len__(self):
        return len(self.images)


class AugmentedDataset(Dataset):
    def __init__(self, augmentations: List[AugmentationConfig], dataset: Dataset, transform=None):
        self.dataset = dataset
        if transform is None:
            self.transform = transforms.Resize((512, 512))
        else:
            self.transform = transform
        self.augmentations = self.create_augmentations_list([IdentityConfig()] + augmentations)
        self.number_of_augmentations = len(augmentations)
        self.len = len(self.dataset) * self.number_of_augmentations

    def __getitem__(self, item):
        correct_idx = item // self.number_of_augmentations
        image, caption = self.dataset[correct_idx]
        augmentation_name, augmentation = self.augmentations[item % self.number_of_augmentations]
        return self.transform(augmentation(image)), augmentation_name

    def __len__(self):
        return self.len

    @staticmethod
    def create_augmentations_list(augmentations: List[AugmentationConfig]) -> List[Tuple[str, Callable]]:
        callables_augs = []
        for aug in augmentations:
            public_members = [attr for attr in dir(aug) if not attr.startswith('_') and not attr == 'augmentation_func']
            name = str(aug)
            func = aug.augmentation_func(**{attr: getattr(aug, attr) for attr in public_members})
            callables_augs.append((name, func))
        return callables_augs


class NoisedCocoCaptions17(CocoCaptions17):
    def __init__(self, num_images_before_noise, num_noise_levels, noise_multiplier, root=datasets_paths["COCO_ROOT"],
                 transform=None):
        super().__init__(root, transform)
        self.num_images_before_noise = num_images_before_noise
        self.num_noise_levels = num_noise_levels
        self.len = num_images_before_noise * num_noise_levels
        self.generator = torch.Generator()
        self.noise_multiplier = noise_multiplier

    def get_consistent_noise(self, idx, cur_noise_level, shape):
        self.generator.manual_seed(idx * cur_noise_level)
        return torch.randn(shape, generator=self.generator) * self.noise_multiplier * \
            (cur_noise_level + 1) / self.num_noise_levels

    def _get_single_item(self, item):
        correct_idx = item // self.num_noise_levels
        image, caption = super()._get_single_item(correct_idx)
        noise = self.get_consistent_noise(correct_idx, item % self.num_noise_levels, image.shape)
        noise_image = torch.clamp(image + noise, 0, 1)
        return noise_image, caption


class NormalDistributedDataset(Dataset):
    def __init__(self, ds_size=500, images_shape=(3, 512, 512), transform=None):
        self.ds_size = ds_size
        self.data = torch.randn(ds_size, *images_shape)
        self.static_caption = "A random noise image"
        self.transform = transform

    def __len__(self):
        return self.ds_size

    def __getitem__(self, item):
        item = self.data[item]
        if self.transform is not None:
            item = self.transform(item)

        return item, self.static_caption


class FolderDataset(Dataset):
    def __init__(self, root, static_caption, transform=None):
        self.root = root
        self.images = os.listdir(root)
        self.len = len(self.images)
        self.static_catpion = static_caption
        if transform is None:
            self.transform = get_default_transform()
        else:
            self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.transform(os.path.join(self.root, self.images[item])), self.static_catpion
