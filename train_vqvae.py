import argparse
import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from model import VQVAE
from dataset import LavalIndoorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data

USE_LMDB = False


def train_vqvae(model: VQVAE,
                device,
                batch_size,
                lr,
                n_epochs,
                l_w_embedding,
                l_w_commitment,
                data_root,
                run_dir='./runs'):
    print('batch size:', batch_size)
    writer = SummaryWriter(run_dir)

    isTrain = True
    data_set = LavalIndoorDataset(data_root=data_root, isTrain=isTrain)
    data_loader = data.DataLoader(data_set, batch_size=batch_size, num_workers=2, shuffle=isTrain)
    L = len(data_loader)

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)

    mse_loss = nn.MSELoss()

    tic = time.time()
    for e in range(1, n_epochs + 1):
        total_loss = 0
        step_base = (e - 1) * L

        for i, (x, _, _) in enumerate(data_loader):
            steps = step_base + i + 1
            current_batch_size = x.shape[0]
            x = x.to(device)

            x_hat, ze, zq = model(x)
            l_reconstruct = mse_loss(x_hat, x)

            l_embedding = mse_loss(ze.detach(), zq)
            l_commitment = mse_loss(ze, zq.detach())

            l_zq = l_w_embedding * l_embedding + l_w_commitment * l_commitment

            loss = l_reconstruct + l_zq

            print(f'step {steps} / epoch:{e} rec_loss: {l_reconstruct.item()}'
                  f'zq_loss:{l_zq.item()}')

            writer.add_scalar('rec_loss ', scalar_value=l_reconstruct.item(), global_step=steps)
            writer.add_scalar('zq_loss ', scalar_value=l_zq.item(), global_step=steps)

            writer.add_scalar('total_loss ', scalar_value=loss.item(), global_step=steps)

            if steps % 100 == 0:
                writer.add_images('GT', torch.clip(torch.exp(x) - 1, 0, 1.), steps)
                writer.add_images('PRED', torch.clip(torch.exp(x_hat) - 1, 0, 1.), steps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size

            print(f'epoch {e} each step loss: {loss.item()}')

        total_loss /= len(data_loader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), f'./checkpoints/vqvae_{tag}_{e}.pth', )

        lr_scheduler.step()

        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')

        writer.add_scalar('Epoch Total Loss', scalar_value=total_loss, global_step=e)
    print('Done')


if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)
    device = f'cuda:0'

    LAVAL_INDOOR_DIR = '/home/xkl/datasets/LavalIndoor'
    img_shape = (3, 128, 128)

    vqvae = VQVAE(img_shape[0], dim=256, n_embedding=256)

    tag = 'codebook128-256_latent32'

    train_vqvae(vqvae,
                device=device,
                batch_size=16,
                lr=5e-4,
                n_epochs=20,
                l_w_embedding=1,
                l_w_commitment=0.25,
                data_root=LAVAL_INDOOR_DIR,
                run_dir=f'./runs_{tag}')
