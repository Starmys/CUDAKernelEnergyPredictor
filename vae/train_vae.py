import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import pandas as pd
import numpy as np


GPU_ID = 1
DEVICE = torch.device(f'cuda:{GPU_ID}')


class Decoder(nn.Module):
    def __init__(self, latent_dims=2, hidden_dims=8, output_dims=6, layers=2):
        super(Decoder, self).__init__()
        self.linear_in = nn.Linear(latent_dims, hidden_dims)
        mid_layers = [nn.Linear(hidden_dims, hidden_dims) for _ in range(layers - 2)]
        self.linear_mid = nn.ModuleList(mid_layers)
        self.linear_out = nn.Linear(hidden_dims, output_dims)

    def forward(self, z):
        z = self.linear_in(z)
        z = F.relu(z)
        for linear_layer in self.linear_mid:
            z = F.relu(z)
            z = linear_layer(z)
        z = self.linear_out(z)
        z = torch.sigmoid(z)
        return z


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims=2, hidden_dims=8, input_dims=6, layers=2):
        super(VariationalEncoder, self).__init__()
        self.linear_in = nn.Linear(input_dims, hidden_dims)
        mid_layers = [nn.Linear(hidden_dims, hidden_dims) for _ in range(layers - 2)]
        self.linear_mid = nn.ModuleList(mid_layers)
        self.linear_mu = nn.Linear(hidden_dims, latent_dims)
        self.linear_sigma = nn.Linear(hidden_dims, latent_dims)
        self.kl = 0

    def forward(self, x):
        x = self.linear_in(x)
        x = F.relu(x)
        for linear_layer in self.linear_mid:
            x = linear_layer(x)
            x = F.relu(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(
            self, latent_dims=2, source_dims=6,
            encoder_hidden_dims=256, decoder_hidden_dims=256,
            encoder_layers=2, decoder_layers=3
        ):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, encoder_hidden_dims, source_dims, encoder_layers)
        self.decoder = Decoder(latent_dims, decoder_hidden_dims, source_dims, decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def save_model(model, name):
    torch.save(model.state_dict(), f'./vae/models/{name}.model')


def train(model, data_train, data_val, epochs=100, loss_balance=10, batch_size=16, val_cycle=50):
    iterations = int(np.ceil(len(data_train) / batch_size))
    print(f'Data size: {data_train.shape}, Batch size: {batch_size}, Iterations: {iterations}')

    opt = torch.optim.Adam(model.parameters())
    beta = 0.02
    loss_list = []
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        np.random.shuffle(data_train)
        sum_loss = np.zeros(2)
        for i in range(iterations):
            x = data_train[i*batch_size:(i+1)*batch_size]
            x = torch.from_numpy(x).to(DEVICE)

            opt.zero_grad()
            x_hat = model(x)
            reconstruction_loss = ((x - x_hat)**2).sum()
            kl_loss = model.encoder.kl
            loss = reconstruction_loss + kl_loss * beta  # 0.00025
            loss.backward()
            opt.step()
            sum_loss += np.array([reconstruction_loss.item(), kl_loss.item()])

            if i % val_cycle == val_cycle - 1:
                avg_loss = sum_loss / val_cycle / batch_size

                x_val = torch.from_numpy(data_val).to(DEVICE)
                x_hat_val = model(x_val)
                reconstruction_loss_val = ((x_val - x_hat_val)**2).sum()
                kl_loss_val = model.encoder.kl
                sum_loss_val = np.array([reconstruction_loss_val.item(), kl_loss_val.item()])
                avg_loss_val = sum_loss_val / data_val.shape[0]

                loss_list.append([avg_loss, avg_loss_val])
                print(f'Iteration {i + 1}, Beta: {"%.5f" % beta}')
                print(f'Avg Loss Train: {"%.4f" % avg_loss[0]}, {"%.1f" % avg_loss[1]}')
                print(f'Avg Loss Val: {"%.4f" % avg_loss_val[0]}, {"%.1f" % avg_loss_val[1]}')
                beta = 0.1 / (1 + np.exp(2 * (np.log(avg_loss[0]) + loss_balance)))
                sum_loss = 0

        if epoch % 50 == 49:
            save_model(model, f'ep-{str(epoch + 1).zfill(4)}')

    print('Finished.')
    save_model(model, 'final')


if __name__ == '__main__':
    torch.manual_seed(2022)
    np.random.seed(2022)

    df = pd.read_csv('./search_space/regnet_convs_unique.csv')
    data = df.to_numpy().astype(np.float32)
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    data_train = data[:4000, :]
    data_val = data[4000:, :]

    latent_dims = 128
    encoder_hidden_dims = 256
    decoder_hidden_dims = 256
    source_dims = 6
    encoder_layers = 5
    decoder_layers = 5

    vae = VariationalAutoencoder(
        latent_dims=latent_dims,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        source_dims=source_dims,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    ).to(DEVICE)
    train(vae, data_train, data_val, epochs=2000, loss_balance=7)
