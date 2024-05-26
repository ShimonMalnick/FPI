import os

import cv2
import numpy as np

from p2p.vis_utils import get_heatmap_and_visualization
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage, ToTensor
from datasets import CocoCaptions17
import torch
from utils import latent_to_bpd

# todo:
# compute the heatmap using an integral image


def compute_heathmap(sample_path, image, out_dir):
    if not os.path.isfile(f"{out_dir}/heatmap.pt"):
        latent = torch.load(sample_path)['latent'].squeeze(0)  # changing the shape from (1, C, H, W) to (C, H, W)
        C, H, W = latent.shape
        kernel_size = 5
        heatmap = torch.zeros(C, H - kernel_size + 1, W - kernel_size + 1)
        for c in range(C):
            for i in range(H - kernel_size + 1):
                for j in range(W - kernel_size + 1):
                    cur_patch = latent[c, i:i + kernel_size, j:j + kernel_size]
                    heatmap[c, i, j] = latent_to_bpd(cur_patch)

        torch.save({'heatmap': heatmap, 'image': image}, f"{out_dir}/heatmap.pt")
    else:
        data = torch.load(f"{out_dir}/heatmap.pt")
        heatmap = data['heatmap']
        image = data['image']

    resized_image = image.resize((heatmap.shape[1] * 8, heatmap.shape[2] * 8))
    resized_image = ToTensor()(resized_image)
    for i in range(heatmap.shape[0]):
        vis, heatmap_mask = get_heatmap_and_visualization(heatmap[i], resized_image)
        heatmap_mask = np.uint8(255 * heatmap_mask)
        heatmap_mask = cv2.cvtColor(np.array(heatmap_mask), cv2.COLOR_RGB2BGR)
        plt.clf()
        fig = plt.figure(figsize=(10, 5))
        cur_ax = fig.add_subplot(1, 3, 1)
        cur_ax.set_title("Heatmap On top of input", fontsize=8)
        im = cur_ax.imshow(vis, vmin=0, vmax=255, cmap='jet')
        cur_ax.axis('off')
        plt.colorbar(im, ax=cur_ax, shrink=0.5)

        cur_ax = fig.add_subplot(1, 3, 2)
        im = cur_ax.imshow(heatmap_mask, vmin=0, vmax=255, cmap='jet')
        plt.colorbar(im, ax=cur_ax, shrink=0.5)
        cur_ax.set_title("BPD Heatmap (higher is less likely)", fontsize=8)
        cur_ax.axis('off')

        cur_ax = fig.add_subplot(1, 3, 3)
        cur_ax.imshow(resized_image.permute(1, 2, 0))
        cur_ax.set_title("Input image", fontsize=8)
        cur_ax.axis('off')

        plt.savefig(f"{out_dir}/heatmap_{i}.jpg")
        plt.close()
        # cv2.imwrite(f"{out_dir}/heatmap_{i}.jpg", vis)

    plt.clf()
    fig = plt.figure()
    cur_ax = fig.add_subplot(1, 2, 1)
    cur_ax.imshow(image)
    cur_ax.axis('off')
    cur_ax.set_title("Original Image", fontsize=8)

    #prepare heat map
    heat_map = heatmap.mean(axis=0)
    heat_map = ToPILImage()(heat_map)
    cur_ax = fig.add_subplot(1, 2, 2)
    im = cur_ax.imshow(heat_map, cmap='jet')
    cur_ax.axis('off')
    cur_ax.set_title("BPD Heatmap (higher is less likely)", fontsize=8)
    plt.colorbar(im, ax=cur_ax, shrink=0.6)

    plt.savefig(f"{out_dir}/heatmap.jpg")
    plt.close()
    # plt.show()


if __name__ == '__main__':
    # base_dir = "/home/shimon/research/diffusion_inversions/FPI/results"
    base_dir = "/home/shimon/research/diffusion_inversions/FPI/results/null_text"

    dataset = CocoCaptions17()
    for i in range(0, 11):
        sample_path = f"{base_dir}/coco/sample_{i}.pt"
        image, caption = dataset[i]
        save_dir = f"{base_dir}/heatmap/sample_{i}"
        os.makedirs(save_dir, exist_ok=True)
        compute_heathmap(sample_path, image, f"{base_dir}/heatmap/sample_{i}")
