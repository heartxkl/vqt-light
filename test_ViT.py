import os
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from einops import rearrange
from torch.utils import data
from dataset import TestImgDataset
import numpy as np
import cv2
from model import VQVAE
import projection_util


BATCH_SIZE_TEST = 1
N_EPOCHS = 25
DEVICE = 'cuda:0'


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, dim * 2)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'Error：the image size cannot be split by patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        self.transformer = Transformer(dim, depth, heads)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        print('init', img.shape)

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        print('rearrange', x.shape)
        x = self.patch_to_embedding(x)
        print('patch_embedding', x.shape)
        print('pos_embedding', self.pos_embedding.shape)
        x += self.pos_embedding
        x = self.transformer(x, mask)
        print('after transformer', x.shape)
        y = self.mlp_head(x)
        print('mlp_head', y.shape)
        return y

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path, map_location='cpu'))
        return self




def evaluate(vit_model, vq_model, data_loader, epoch, result_dir):
    vit_model.eval()

    os.makedirs(f'{result_dir}', exist_ok=True)

    L = len(data_loader)

    with torch.no_grad():
        for i, (photo, img_id) in enumerate(data_loader):

            photo = photo.to(DEVICE)

            y = vit_model(photo).view((-1, VOC_SIZE))  # (Bx256, 64)

            output = F.log_softmax(y, dim=1)  # (Bx256, 64)

            if not isTrain:
                _, pred = torch.max(output, dim=1)

                pred_indices = pred.view((-1, Latent_Size, Latent_Size))

                hdr_preds = vq_model.decode(pred_indices)

                hdr_preds = torch.exp(hdr_preds) - 1
                hdr_preds = hdr_preds.cpu().numpy()
                B = hdr_preds.shape[0]

                for bi in range(B):
                    hdr_pred = hdr_preds[bi]
                    hdr_pred = np.transpose(hdr_pred, (1, 2, 0))
                    hdr_pred = hdr_pred[:, :, [2, 1, 0]]

                    hdr_path = f'./{result_dir}/{img_id[bi]}.exr'
                    hdr_name = f'/{img_id[bi]}.exr'
                    cv2.imwrite(hdr_path, hdr_pred)

                    hdr_pred = np.clip(hdr_pred, 0, 1.0)
                    cv2.imwrite(f'./{result_dir}/{img_id[bi]}.png', np.uint8(hdr_pred * 255))

                    ep_save_folder = result_dir + '/SP2EP'
                    os.makedirs(ep_save_folder, exist_ok=True)
                    projection_util.SP2EP(hdr_path, hdr_name=hdr_name, save_path=ep_save_folder)


isTrain = False
torch.manual_seed(42)
Latent_Size = 32
VOC_SIZE = 128

tag = 'codebook128-256_latent32'

test_set = TestImgDataset(input_dir='./test_photos')
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=2)

vit_model = ViT(image_size=256, patch_size=8, num_classes=128, channels=3, dim=256, depth=6, heads=8, mlp_dim=256)

vit_model.to(DEVICE)

optimizer = optim.Adam(vit_model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15, 20], gamma=0.1)

vq_model = VQVAE(3, 256, 128)
vq_model.load_model('./checkpoints/vqvae_codebook128-256_latent32_20.pth')
vq_model.to(DEVICE)
vq_model.eval()

last_epoch = 25
vit = vit_model.load_model(f'./checkpoints/vit_{tag}_{last_epoch}.pth')
vit.to(DEVICE)
total_sce = 0
for epoch in range(1, 2):
    evaluate(vit, vq_model, test_loader, epoch, f'results_{tag}')