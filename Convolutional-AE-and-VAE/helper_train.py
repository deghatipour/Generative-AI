import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_conv_autoencoder(model, train_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, _ in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            reconstructions = model(inputs)
            loss_function = nn.MSELoss()            
            loss = loss_function(reconstructions, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print('Epoch: %03d/%03d | Loss: %.4f'
                % (epoch+1, num_epochs, avg_epoch_loss))






def loss_function(x, x_reconstructed, z_mean, z_logvar):
    reproduction_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    KLD = KLD.mean()

    return reproduction_loss + KLD


def train_conv_vae(model, train_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, _ in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            z_mean, z_logvar, decoded = model(inputs)
            loss = loss_function(inputs, decoded, z_mean, z_logvar)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        avg_loss = epoch_loss / len(train_loader)
        print('Epoch: %03d/%03d | Loss: %.4f'
                % (epoch+1, num_epochs, avg_loss))

