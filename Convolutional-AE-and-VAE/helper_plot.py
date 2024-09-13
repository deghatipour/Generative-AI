
import torch
import numpy as np
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_conv_autoencoder_reconstructions(model, loader, n_images = 5):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images.to(device)
        reconstructions = model(images[:n_images]).cpu().numpy()
        reconstructions = np.clip(reconstructions, 0, 1)
        images = images.cpu().numpy()
        images = np.clip(images, 0, 1)

    plt.figure(figsize = (n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        images = images.reshape(-1, 28, 28)
        plt.imshow(images[image_index], cmap = "binary")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        reconstructions = reconstructions.reshape(-1, 28, 28)
        plt.imshow(reconstructions[image_index], cmap = "binary")
    plt.show()




def plot_vae(model, lodaer, n_images = 5):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(lodaer))
        images = images.to(device)
        reconstruction = model(images[:n_images])[2].cpu().numpy()
        reconstruction = np.clip(reconstruction, 0, 1)
        images = images.cpu().numpy()
        images = np.clip(images, 0, 1)

    plt.figure(figsize = (n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        images = images.reshape(-1, 28, 28)
        plt.imshow(images[image_index], cmap = "binary")
        plt.axis("off")
        
        plt.subplot(2, n_images, 1 + n_images + image_index)
        reconstruction = reconstruction.reshape(-1, 28, 28)
        plt.imshow(reconstruction[image_index], cmap = "binary")
        plt.axis("off")
    plt.show()
