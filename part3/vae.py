import torch
from torch import nn
from torch.autograd import Variable

# Class for encoder
class Encoder(nn.Module):
    # Init method
    def __init__(self, input_dim, z_dim):
        super().__init__()
        # CNN layers
        self.enc = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=(4,4), stride=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=(4,4), stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 16, kernel_size=(4,4), stride=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=(4,4), stride=2),
        nn.ReLU()
        )
        # mu and var layers
        self.mu = nn.Conv2d(32, 32, kernel_size=(1,1), stride=1) # mu layer
        self.var = nn.Conv2d(32, 32, kernel_size=(1,1), stride=1) # logvariance layer
    # Forward method
    def forward(self, x):
        hidden = self.enc(x)
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)

        return z_mu, z_var

# Class for decoder
class Decoder(nn.Module):
    # Init method
    def __init__(self, z_dim, output_dim, hidden_dim=100):
        super().__init__()
        # CNN layers
        self.dec = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=1),
            torch.nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=1),
            torch.nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=1),
            torch.nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, 3, kernel_size=(4, 4), stride=1),
            torch.nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(3, 3, kernel_size=(4, 4), stride=1),
            torch.nn.Upsample(size=(64, 64))
        )
    # Forward method
    def forward(self, x):
        predicted = self.dec(x)

        return predicted

# Class for Variational Auto Encoder
class VAE(nn.Module):
    # Init method
    def __init__(self, input_dim, z_dim):
        output_dim = input_dim
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, z_dim)
        self.decoder = Decoder(z_dim, output_dim)
    # Encode method
    def encode(self, x):
        mu,var = self.encoder(x)
        return mu, var
    # Decode method
    def decode(self, z):
        predicted = self.decoder(z)
        return predicted
    # Reparameterization - adding epsilon
    def reparameterize(self, mu, logvar):
        # mu is mean and logvar is variance
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    # forward method
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar