
import torch
import torchvision


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_gan(model, 
             train_loader, 
             optimizer_generator, 
             optimizer_discriminator, 
             num_epochs, 
             latent_dim
             ):

    results_dict = {'images_from_noise_per_epoch': []}

    loss = torch.nn.functional.binary_cross_entropy_with_logits
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device = DEVICE)

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (real, _ ) in enumerate(train_loader):
            batch_size = real.size(0)

            # Real images    
            real_images = real.to(DEVICE)
            real_labels = torch.ones(batch_size, device = DEVICE)

            # Generated (Fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device = DEVICE)
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device = DEVICE)
            flipped_fake_labels = real_labels


            # -----------------------------------------------------
            # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            # -----------------------------------------------------
            optimizer_discriminator.zero_grad()

            # Discriminator loss on real images
            discriminator_pred_real = model.discriminator_forward(real_images).view(-1)
            loss_real = loss(discriminator_pred_real, real_labels)

            # Discriminator loss on fake images
            discriminator_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            loss_fake = loss(discriminator_pred_fake, fake_labels)

            loss_discriminator = 0.5 * (loss_real + loss_fake)
            loss_discriminator.backward()

            optimizer_discriminator.step()


            # ---------------------------------------------------------
            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # ---------------------------------------------------------
            optimizer_generator.zero_grad()

            discriminator_pred_fake = model.discriminator_forward(fake_images).view(-1)
            loss_generator = loss(discriminator_pred_fake, flipped_fake_labels)
            loss_generator.backward()
            optimizer_generator.step()


            # --------------------------
            # Results
            # --------------------------   
            if batch_idx % 100 == 0:
                print(
                    f'Epoch [{epoch + 1 }/{num_epochs}] \
                    Loss D: {loss_discriminator:.4f}, loss G: {loss_generator:.4f}'
                )

        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise).detach().cpu()
            results_dict['images_from_noise_per_epoch'].append(torchvision.utils.make_grid(fake_images, padding = 2, normalize = True))

    return results_dict