import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size):
    net=[nn.Linear(input_size, size),nn.ReLU()]

    for _ in range(n_layers):
        net.append(nn.Linear(size,size))
        net.append(nn.ReLU())
    net.append(nn.Linear(size, output_size))
    return nn.Sequential(*net)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
