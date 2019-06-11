import torch
import torch.nn.functional as F

class BCE_KL_loss(torch.nn.Module):
    """ 
    Reconstruction loss for variational auto-encoders.
    Binary-cross entropy reconstruction + KL divergence losses summed
    over all elements and batch. 
    Mostly taken from pytorch examples: 
        https://github.com/pytorch/examples/blob/master/vae/main.py

    Arguments:
        outputs: List of the form [reconstruction, mean, logvariance].
        x: ground-truth.
    """
    def __init__(self):
        super(BCE_KL_loss, self).__init__()

    def forward(self, outputs, target):
        recon_x, mu, logvar = outputs
        BCE = F.binary_cross_entropy(recon_x, target, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

class MSE_KL_loss(torch.nn.Module):
    """ 
    Reconstruction loss for variational auto-encoders.
    Mean squared error reconstruction + KL divergence losses summed
    over all elements and batch. 
    Mostly taken from pytorch examples: 
        https://github.com/pytorch/examples/blob/master/vae/main.py

    Arguments:
        outputs: List of the form [reconstruction, mean, logvariance].
        x: ground-truth.
    """
    def __init__(self):
        super(MSE_KL_loss, self).__init__()

    def forward(self, outputs, target):
        recon_x, mu, logvar = outputs
        MSE = F.mse_loss(recon_x, target, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

class Multihead_loss(torch.nn.Module):
    """
    Compute the loss on multiple outputs.

    Arguments:
        outputs: List of network outputs.
        target: List of targets where len(outputs) = len(target).
        loss_function: List of loss functions with either
            len(loss_function) = len(targets) or len(loss_function) = 1.
        weights: List of weights for each loss. Default = [1]
    """
    def __init__(self, loss_function, weights=[1]):
        super(Multihead_loss, self).__init__()

        self.loss_function = loss_function
        self.weights = weights

    def forward(self, outputs, target):
        assert(len(outputs) == len(target))
        assert(len(self.loss_function) == len(target) \
            or len(self.loss_function) == 1)

        # expand loss_function list if univariate
        if len(self.loss_function) == 1:
            self.loss_function = [self.loss_function[0] for i in range(len(target))]
        # expand weights list if univariate
        if len(self.weights) == 1:
            self.weights = [self.weights[0] for i in range(len(target))]

        # compute loss for each head
        total_loss = 0.
        for out, gt, loss_func, weight in zip(outputs, target, self.loss_function, self.weights):
            loss = loss_func(out, gt)
            total_loss += loss * weight
        return total_loss
