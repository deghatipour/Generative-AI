
import numpy as np
import torchvision
from matplotlib import pyplot as plt



def plot_images(data_loader):
    images, _ = next(iter(data_loader))
    plt.figure(figsize = (8, 8))
    plt.axis("off")
    plt.title("Images in Training Dataset")
    plt.imshow(np.transpose(torchvision.utils.make_grid(images[:64], padding = 2, normalize = True),(1, 2, 0)))
    plt.show()



def plot_generated_images(NUM_EPOCHS, log_dict):
    for i in range(0, NUM_EPOCHS, 5):
        plt.figure(figsize = (8, 8))
        plt.axis('off')
        plt.title(f'Generated Images at Epoch {i}')
        plt.imshow(np.transpose(log_dict['images_from_noise_per_epoch'][i], (1, 2, 0)))
        plt.show()

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Generated Images after Last Epoch')
    plt.imshow(np.transpose(log_dict['images_from_noise_per_epoch'][-1], (1, 2, 0)))
    plt.show()