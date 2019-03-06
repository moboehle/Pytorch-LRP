import torch


def pprint(*args):
    out = [str(argument) + "\n" for argument in args]
    print(*out, "\n")


class Identity(torch.nn.Module):
    # Maybe deprecated... Initially meant for gradient in input layer
    # Not useful anymore?
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(layer_input: torch.Tensor):
        fake_mul = torch.Tensor([1])
        fake_mul.requires_grad_()
        return fake_mul * layer_input

    @staticmethod
    def backward(gradient_input):
        return gradient_input


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))
