import argparse

from torchvision import datasets, transforms

parser = argparse.ArgumentParser('MNIST classification arguments')

parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
parser.add_argument("--batch_size", type=int, default=1000, help="Mini-Batch size")
parser.add_argument("--num_epochs_pretrain", type=int, default=1000, help="Number of pre-training epochs")
parser.add_argument("--num_epochs_classify", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lr_decoder_pretrain", type=float, default=0.001, help="Learning rate for decoder")
parser.add_argument("--lr_encoder_pretrain", type=float, default=0.001,
                    help="Learning rate for encoder during pre-training")
parser.add_argument("--lr_encoder_classify", type=float, default=0.0001,
                    help="Learning rate for encoder when fine-tuning for classification")
parser.add_argument("--save_epochs", type=int, default=20,
                    help="Number of epochs after which to save the network periodically")
parser.add_argument("--dataset", type=str, default="MNIST", help="name of dataset")


def get_mnist_parser():
    return parser


def get_logfile_name(args):
    logfile = ""

    logfile += f"{args.dataset}_batch_size={args.batch_size}"
    logfile += f"_seed={args.seed}_num_epochs_pretrain={args.num_epochs_pretrain}_classify_{args.num_epochs_classify}"
    logfile += f"_lr_decoder_pre_{args.lr_decoder_pretrain}_encoder_pre_{args.lr_encoder_pretrain}"
    logfile += f"_classify_{args.lr_encoder_classify}_other"
    return logfile


def get_dataset(dataset_name):
    if dataset_name == "MNIST":
        train_data = datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

        test_data = datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        return train_data, test_data
    elif dataset_name == "FashionMNIST":
        train_data = datasets.FashionMNIST(root="./data/fashionmnist", train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))

        test_data = datasets.FashionMNIST(root="./data/fashionmnist", train=False, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor()
                                          ]))
        return train_data, test_data
