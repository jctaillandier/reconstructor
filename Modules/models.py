from torch import nn
import torch
from torch.nn import functional as F

class Autoencoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Linear(in_dim, in_dim),
            # nn.LeakyReLU(),
            nn.Linear(in_dim,out_dim),
            nn.LeakyReLU(), 
            nn.Linear(out_dim,out_dim),
            nn.LeakyReLU()
        )

    def forward(self, xin):
        x = self.encoder(xin)
        return x.cpu().numpy()

class VAE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(in_dim, 50)
        self.fc21 = nn.Linear(50, 20)  # for mu
        self.fc22 = nn.Linear(50, 20)  # for std_dev
        self.fc3 = nn.Linear(20, 50)
        self.fc4 = nn.Linear(50, out_dim)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        return F.leaky_relu(self.fc4(h3))

    def forward(self, xin):
        mu, logvar = self.encode(xin)
        x = self.reparameterize(mu, logvar)
        return self.decode(x), mu, logvar


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))