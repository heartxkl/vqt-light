import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from model import VQVAE
from dataset import LavalIndoorDataset
from torch.utils import data


# visualizing reconstruction result
def reconstruct(model, x):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)
    n = x.shape[0]
    n1 = int(n ** 0.5)
    x_cat = torch.cat((x, x_hat), 3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'work_dirs/vqvae_reconstruct.jpg', x_cat)


if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)
    device = f'cuda:0'

    img_shape = (3, 128, 128)

    tag = 'codebook128-256_latent32'

    vqvae = VQVAE(img_shape[0], dim=256, n_embedding=128)

    last_epoch = 20
    vqvae.load_model(f'./checkpoints/vqvae_{tag}_{last_epoch}.pth')

    data_set = LavalIndoorDataset(data_root='/home/xkl/datasets/HavenIndoor', isTrain=True)
    data_loader = data.DataLoader(data_set, batch_size=100, num_workers=2, shuffle=False)

    img = next(iter(data_loader))[0].to(device)
    reconstruct(vqvae, img)
