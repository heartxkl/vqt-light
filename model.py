
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


class VQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding):
        assert dim >= 128 and dim % 4 == 0
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, 128, 4, 2, 1),  # (N, 128, 64, 64)
                                     nn.ReLU(),

                                     nn.Conv2d(128, 256, 4, 2, 1),  # (N, 256, 32, 32)
                                     nn.ReLU(),

                                     ResidualBlock(dim),
                                     ResidualBlock(dim))
        # codebook
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # (N, 128, 64, 64)
            ResidualBlock(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # (N, 32, 128, 128)
            ResidualBlock(32),
            nn.ReLU(),

            nn.Conv2d(32, input_dim, 3, 1, 1))  # (N, 3, 128, 128)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.n_downsample = 3

    def forward(self, x):
        # encode
        ze = self.encoder(x)

        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        # (N, H=16, W=16)
        nearest_neighbor = torch.argmin(distance, 1)
        # make C to the second dim (N, H=32, W=32, C=128) -> (N, C=128, H=32, W=32)
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)  # get zq by indexing codebook
        # stop gradient
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq

    @torch.no_grad()
    def encode(self, x):
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor

    @torch.no_grad()
    def decode(self, discrete_latent):
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat

    # Shape: [C, H, W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        # 128 // 2^3 = 16
        return (H // 2 ** self.n_downsample, W // 2 ** self.n_downsample)

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path, map_location='cpu'))
        return self


if __name__ == '__main__':
    model = VQVAE(input_dim=3, dim=256, n_embedding=256)
    torch.save(model.state_dict(), './vqvae_latent32_embdim256_vocsize256.pth')
    x = torch.randn((2, 3, 128, 128))

    # model.load_state_dict(torch.load('./xxx.pth', map_location='cpu'))
    x_hat, _, _ = model(x)
    print(x_hat.shape)
    indices = model.encode(x)
    print(indices.shape)
