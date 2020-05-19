import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


# Redefine function such as nllloss if you wish to use them as loss. See the example below
# Note that the shape of the given input is [mxn] and the target is [m] (just as for NLLLoss).
# See Accuracy for example to see how to handle the specific format.

class NLLLoss(nn.NLLLoss):
    """
    Redefinition of NLLLoss, just to take into account specificities in the forward pass
    """

    def __init__(self, *args, **kwargs):
        """
        Take exactly the same arguments as the nn.NLLLoss (see:
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html)
        """
        super(NLLLoss, self).__init__(*args, **kwargs)

    def forward(self, input_, target, *args, **kwargs):
        """
        Exactly as the nll loss, we simply ignore the other attributes
        """
        return super().forward(input_, target)


class AccuracyLoss(nn.Module):
    """
    Compute the accuracy loss, in a compatible way as nn.Module (Mostly for CUDA)
    """

    def __init__(self, device="cpu"):
        super(AccuracyLoss, self).__init__()

    def forward(self, input_, target, *args, **kwargs):
        """
        Compute the accuracy
        """
        # Selecting the outputs that should be maximized (equal to 1)
        out = input_[range(len(target)), target]
        # computing accuracy loss (distance to 1
        return 1 - out.mean()


class MSELoss(nn.MSELoss):
    """
    Wrapper around the torch mse loss to facilitate integration with other custom losses
    """

    def __init__(self, device="cpu"):
        super(MSELoss, self).__init__()

    def forward(self, input_, target, *args, **kwargs):
        """ Just ignore args and kwargs"""
        return super().forward(input_, target)


class CircularAccuracyLoss(nn.Module):
    """
    Compute a circular accuracy loss: the target is bins away from the real target value. If a value is above a maximum,
    then we map it back to the minimum value.
    """

    def __init__(self, max_=3, step=2, device="cpu"):
        """
        Initialization
        :param max_: the maximum value of all possible targets
        :param step: the difference between the real value and the wished target
        :param device: the device on which to compute the results.
        """
        super(CircularAccuracyLoss, self).__init__()
        self.max_ = max_ + 1
        self.step = step
        self.device = device
        self.fn = AccuracyLoss(device)

    def forward(self, input_, target, *args, **kwargs):
        """
        Compute the loss
        :param input_: the predicted values
        :param target: the real value without any modifications, straight out of the dataset
        """
        t = (target + self.step) % self.max_
        t.to(self.device)
        return self.fn(input_, t, *args, **kwargs)


class BalancedErrorRateLoss(nn.Module):
    """
    Compute the balanced error rate loss.
    """

    def __init__(self, targetBer=1 / 2, device="cpu"):
        """
        :param targetBer: the value of the BER to be closed to
        """
        super(BalancedErrorRateLoss, self).__init__()
        self.targetBer = targetBer
        self.device = device

    def get_true_value(self, computed_ber):
        """
        Return the true value of the computed BER, not the distance from target. BER is between [0,1/2]. Impossible to
        go beyond that interval unless wrongly implemented.
        :param computed_ber: the distance from the target BER. Must be of type numpy array.
        """
        if isinstance(computed_ber, list):
            computed_ber = np.array(computed_ber)
        b = -computed_ber + self.targetBer
        return np.abs(b)

    def forward(self, input_, target, group, *args, **kwargs):
        """
        Comput the balanced error rate
        :param input_: the input
        :param target: the target values
        :param group: the attribute whose values allow group constructions
        :param args: the argument to ignore
        :param kwargs: the keyword arguments to ignore
        :return: the computed loss
        """
        # Selecting the right predictions,
        out = input_[range(len(target)), target]
        # Computing errors
        out = torch.abs(1 - out)
        # Reshaping data
        group.to(self.device)
        group = group.view(-1)
        out = out.view(-1, 1)
        l = len(out)
        # Summing by mask values
        # From: https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
        m = torch.zeros(group.max() + 1, l).to(self.device)
        m[group, torch.arange(l)] = 1
        m = nn.functional.normalize(m, p=1, dim=1)
        # Computing the mean of each group\
        out = torch.mm(m, out)
        out = out.mean()
        # k = torch.abs(input.argmax(1) - target).type(torch.FloatTensor).to(self.device)
        # k = (k[sens==1].mean() + k[sens==0].mean())/2
        return torch.abs(self.targetBer - out)


class MaeSacLoss(nn.Module):
    """
    Mae Sac loss. First loss is the reconstruction Loss, second is the sensitive attribute one.
    """

    def __init__(self, alpha_=0.25, device="cpu", ae_reduction="mean", *args, **kwargs):
        """
        Initialisation
        :param alpha_: the coefficient for fairness. reconstruction will be (1-alpha_)
        :param device: the device on which to compute the loss
        :param ae_reduction: the type of computation wanted for the reconstruction loss. Either sum, mean or none.
        :param ber_target: the target of the sensitive attribute loss.
        """
        super(MaeSacLoss, self).__init__()
        self.post_rec = lambda x: x
        self.rec_loss = nn.L1Loss(reduction=ae_reduction)
        if ae_reduction == "none":
            self.post_rec = lambda x: x.mean(0)
        self.group_loss = AccuracyLoss(device=device)
        self.alpha_ = alpha_

    def forward(self, data_, data_target, disc_pred, disc_target, group, *args, **kwargs):
        rec = self.rec_loss(data_, data_target)
        rec = self.post_rec(rec)
        gr = self.group_loss(disc_pred, disc_target, group)
        return (1 - self.alpha_) * rec, self.alpha_ * gr


class MaeBerLoss(nn.Module):
    """
    Mae Sac loss. First loss is the reconstruction Loss, second is the sensitive attribute one.
    """

    def __init__(self, alpha_=0, device="cpu", ae_reduction="mean", ber_target=1 / 2, *args, **kwargs):
        """
        Initialisation
        :param alpha_: the coefficient for fairness. reconstruction will be (1-alpha_)
        :param device: the device on which to compute the loss
        :param ae_reduction: the type of computation wanted for the reconstruction loss. Either sum, mean or none.
        :param ber_target: the target of the sensitive attribute loss.
        """
        super(MaeBerLoss, self).__init__()
        self.post_rec = lambda x: x
        self.rec_loss = nn.L1Loss(reduction=ae_reduction)
        if ae_reduction == "none":
            self.post_rec = lambda x: x.mean(0)
        # self.group_loss = BalancedErrorRateLoss(targetBer=ber_target, device=device)
        self.alpha_ = alpha_

    def forward(self, data_, data_target, *args, **kwargs):
        rec = self.rec_loss(data_, data_target)
        rec = self.post_rec(rec)
        # gr = self.group_loss(disc_pred, disc_target, group)
        # First loss is the reconstruction Loss, second is the sensitive attribute one.
        return (1 - self.alpha_) * rec #, self.alpha_ * gr



# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss(recon_x, x, mu, logvar):
    loss_fn = torch.nn.L1Loss(reduction='none')
    l1_loss = loss_fn(recon_x, x,reduction='none')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD = (KLD-mu)/logvar

    # import pdb;pdb.set_trace()
    return l1_loss + KLD