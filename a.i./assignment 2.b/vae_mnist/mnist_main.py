import os

import numpy as np
import torch
import torch.nn.functional as torchfunc
from tensorboardX import SummaryWriter
from torch.distributions import kl_divergence, Normal
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from vae_mnist.mnist_parser import get_mnist_parser, get_logfile_name, get_dataset
from vae_mnist.vae import VAE

parser = get_mnist_parser()
args = parser.parse_args()

modeldir = "./model/"
os.makedirs(modeldir, exist_ok=True)

logdir = "./tensorboard_log/"
logfile = get_logfile_name(args)

writer = SummaryWriter(logdir + logfile)

torch.manual_seed(args.seed)

train_data, test_data = get_dataset(args.dataset)
train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)

train_dataset_size = len(train_data)
test_dataset_size = len(test_data)

print(f"\nDataset size - Train -- {train_dataset_size}, Test -- {test_dataset_size}")

print(f"Image size {train_data[0][0].size()}")

print("\nStarted pre training")

encoder_layer_sizes = [28 * 28, 400, 100]
decoder_layer_sizes = [100, 400, 28 * 28]
bottleneck_layer_size = 20
vae = VAE(encoder_layer_sizes=encoder_layer_sizes, decoder_layer_sizes=decoder_layer_sizes,
          bottleneck_layer_size=bottleneck_layer_size, activation_function=torchfunc.relu)
encoder_params = list(vae.encoder.parameters()) + list(vae.mean_layer.parameters()) + list(
    vae.log_std_layer.parameters())
encoder_optimizer = Adam(encoder_params, lr=args.lr_encoder_pretrain)
decoder_optimizer = Adam(vae.decoder.parameters(), lr=args.lr_decoder_pretrain)

global_step = 1
best_model_loss = np.inf

for i in range(10):
    writer.add_image(f'Image_{i+1}_ground_truth', train_data[i][0].view(28, 28), 0)

for epoch in range(args.num_epochs_pretrain):
    epoch_loss = 0
    for batch_id, (data, labels) in enumerate(train_data_loader):
        data = data.view(args.batch_size, 28 * 28)
        x_recon_batch, z_batch, mean, std = vae(data)
        recon_loss = torchfunc.binary_cross_entropy(x_recon_batch, data, reduction='sum')
        prior = Normal(0, 1)
        kl_divergence_loss = 0
        for i in range(args.batch_size):
            encoder_distribution = Normal(mean[i], std[i])
            kl_divergence_loss += kl_divergence(encoder_distribution, prior).sum()

        total_loss = kl_divergence_loss + recon_loss

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        kl_divergence_loss.backward(retain_graph=True)
        recon_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        writer.add_scalar("KL divergence", kl_divergence_loss, global_step)
        writer.add_scalar("Reconstruction loss", recon_loss, global_step)
        global_step += 1
        epoch_loss += total_loss
    writer.add_scalar("Epoch loss", epoch_loss, epoch)
    print(f"Epoch {epoch} completed. Loss = {epoch_loss}")

    if epoch_loss < best_model_loss:
        print(f"Found new best model")
        torch.save(vae.state_dict(), modeldir + "best_vae.pt")

    if epoch % args.save_epochs == 0:
        print(f"Saving periodically for epoch {epoch}")
        torch.save(vae.state_dict(), modeldir + "periodic_vae.pt")

    for i in range(10):
        image, _, _, _ = vae(train_data[i][0].view(1, 28 * 28))
        writer.add_image(f'Image_{i+1}', image.view(28, 28), epoch)

print("Finished pre training")
