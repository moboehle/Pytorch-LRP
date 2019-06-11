# Initialize weights
from torch.nn import init, Conv3d, BatchNorm3d, Linear


def xavier(x):
    return init.xavier_normal_(x)

def xavier_uniform(x):
    return init.xavier_uniform_(x)

def he(x):
    return init.kaiming_normal_(x)

def he_uniform(x):
    return init.kaiming_uniform_(x)


def weights_init(m, func=he_uniform):
    if isinstance(m, Conv3d):
        func(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, BatchNorm3d):
        m.reset_parameters()
    elif isinstance(m, Linear):
        m.reset_parameters()
