import json
import os
from pathlib import Path
from typing import Union, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageNet
from setup import setup_config

datasets_paths = {"COCO_ROOT": "mscoco17",
                  "CHEST_XRAY_ROOT": "chest_xray",
                  "IMAGENET_ROOT": "imagenet"}

datasets_paths = {k: os.path.join(setup_config['DATASETS_ROOT'], v) for k, v in datasets_paths.items()}


def get_default_transform():
    return transforms.Compose([lambda image: Image.open(image).convert('RGB').resize((512, 512)), transforms.ToTensor()])


class ImageNetSubset(ImageNet):
    def __init__(self, root: Union[str, Path] = datasets_paths['IMAGENET_ROOT'], split: str = 'train', num_classes=100,
                 num_images_per_class=100, **kwargs: Any):
        super().__init__(root, split, **kwargs)
        self._num_classes = num_classes
        self._num_images_per_class = num_images_per_class


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


class NormalDistributedDataset(Dataset):
    def __init__(self, ds_size=500, images_shape=(3, 512, 512), transform=None):
        self.ds_size = ds_size
        self.data = torch.randn(ds_size, *images_shape)
        self.static_caption = "A random noise image"
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return self.ds_size

    def __getitem__(self, item):
        return self.transform(self.data[item]), self.static_caption


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
