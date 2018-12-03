import torch
import torch.nn as nn
from torch.distributions import Normal


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, decoder_layer_sizes, bottleneck_layer_size, activation_function,
                 bias=False):
        super().__init__()
        self.bias = bias
        self.activation_function = activation_function

        decoder_layer_sizes.insert(0, bottleneck_layer_size)

        self.encoder = nn.ModuleList([nn.Linear(encoder_layer_sizes[i], encoder_layer_sizes[i + 1], bias=bias)
                                      for i in range(len(encoder_layer_sizes) - 1)])

        self.mean_layer = nn.Linear(encoder_layer_sizes[-1], bottleneck_layer_size, bias=bias)
        self.log_std_layer = nn.Linear(encoder_layer_sizes[-1], bottleneck_layer_size, bias=bias)

        self.decoder = nn.ModuleList([nn.Linear(decoder_layer_sizes[i], decoder_layer_sizes[i + 1], bias=bias)
                                      for i in range(len(decoder_layer_sizes) - 1)])

    def encode(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x = self.activation_function(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        return mean, torch.exp(log_std)

    def decode(self, x):
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
            x = self.activation_function(x)
        return torch.sigmoid(x/10)

    def forward(self, x):
        mean, std = self.encode(x)
        normal = Normal(mean, std)
        z = normal.rsample()
        x_recon = self.decode(z)
        return x_recon, z, mean, std
