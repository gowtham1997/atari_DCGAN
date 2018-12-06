import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_shape, filters):
        super().__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],
                      out_channels=filters,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters,
                      out_channels=filters * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters * 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters * 2,
                      out_channels=filters * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters * 4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters * 4,
                      out_channels=filters * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters * 8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters * 8,
                      out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv_pipe(x).view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape, filters, latent_vector_dim):
        super().__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_vector_dim,
                               out_channels=filters * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(filters * 8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=filters * 8,
                               out_channels=filters * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters * 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=filters * 4,
                               out_channels=filters * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters * 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=filters * 2,
                               out_channels=filters,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=filters,
                               out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)
