import argparse


def get_args():
    parser = argparse.ArgumentParser("temporal coding")

    parser.add_argument("--tau", default=1, type=float, help="tau in neuron")
    parser.add_argument("--t_scale", default=6, type=float, help="scale in time")
    parser.add_argument("--dataset", default='fmnist', type=str, help="dataset")

    parser.add_argument("--epochs", default=100, type=int, help="epoch")
    parser.add_argument("--batch_size", default=64, type=int, help="epoch")
    parser.add_argument("--lr", default=0.0006, type=float, help="float")

    parser.add_argument("--arch", default='shuffle', type=str, help="[base, res, shuffle]")
    parser.add_argument("--t_init", default=0.5, type=float, help="float")


    args = parser.parse_args()

    return args
