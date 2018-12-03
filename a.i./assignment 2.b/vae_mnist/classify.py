import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from vae_mnist.mnist_parser import get_mnist_parser, get_logfile_name, get_dataset
from vae_mnist.vae import VAE

parser = get_mnist_parser()
args = parser.parse_args()

modeldir = "./model/classify/"
os.makedirs(modeldir, exist_ok=True)

logdir = "./tensorboard_log/classify/"
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

print("\nStarted training")

encoder_layer_sizes = [28 * 28, 400, 100]
decoder_layer_sizes = [100, 400, 28 * 28]
bottleneck_layer_size = 20
vae = VAE(encoder_layer_sizes=encoder_layer_sizes, decoder_layer_sizes=decoder_layer_sizes,
          bottleneck_layer_size=bottleneck_layer_size, activation_function=torchfunc.relu)
encoder_params = list(vae.encoder.parameters()) + list(vae.mean_layer.parameters()) + list(
    vae.log_std_layer.parameters())

# load model
vae.load_state_dict(torch.load("./model/best_vae.pt"))
final_layer = nn.Linear(20, 10, False)

encoder_optimizer = Adam(encoder_params, lr=args.lr_encoder_classify)
final_optimiser = Adam(final_layer.parameters(), lr=0.001)
global_step = 1
best_model_loss = np.inf

for epoch in range(args.num_epochs_pretrain):
    epoch_loss = 0
    for batch_id, (data, labels) in enumerate(train_data_loader):
        data = data.view(args.batch_size, 28 * 28)
        x_mean, _ = vae.encode(data)
        x = final_layer(x_mean)
        y_pred = torchfunc.softmax(x, dim=1)

        classification_loss = torchfunc.nll_loss(y_pred,labels)

        encoder_optimizer.zero_grad()
        final_optimiser.zero_grad()
        encoder_optimizer.step()
        final_optimiser.step()

        writer.add_scalar("Classification loss", classification_loss, global_step)

        pred = y_pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().item()

        writer.add_scalar("Accuracy", correct/args.batch_size, global_step)
        global_step += 1

    # if epoch_loss < best_model_loss:
    #     print(f"Found new best model")
    #     torch.save(vae.state_dict(), modeldir + "best_vae.pt")
    #
    # if epoch % args.save_epochs == 0:
    #     print(f"Saving periodically for epoch {epoch}")
    #     torch.save(vae.state_dict(), modeldir + "periodic_vae.pt")
    #
    # for i in range(10):
    #     image, _, _, _ = vae(train_data[i][0].view(1, 28 * 28))
    #     writer.add_image(f'Image_{i+1}', image.view(28, 28), epoch)

print("training")
