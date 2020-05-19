import torch
import numpy as np
import torch.nn as nn


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



class DamageAttributeLoss(nn.Module):
    """ Compute the loss per attribute and return it as a vector. """

    def __init__(self, categorical_indexes, numerical_indexes, hard=False, alpha_=0.25, device="cpu", ber_target=1 / 2,
                 *args, **kwargs):
        """
        Initialisation. We assume that for numerical columns, we will use the l1-loss. Change accordingly
        :param categorical_indexes: indexes of the categorical columns in list of dict
        :param numerical_indexes: index of numerical columns as list.
        :param hard: Should we be hard on the categorical error or should we just maximise the right element ?
        See hard_categorical for a more in depth explanation
        :param alpha_: the coefficient for fairness. reconstruction will be (1-alpha_)
        :param device: the device on which to compute the loss
        :param ber_target: the target of the sensitive attribute loss.
        """
        super(DamageAttributeLoss, self).__init__()
        self.l1_none = nn.L1Loss(reduction="none")
        self.c_l1_mean = lambda x, target: self.l1_none(x, target).mean(0)
        self.cat_indexes = categorical_indexes
        self.num_indexes = numerical_indexes
        self.hard = hard
        if self.hard:
            self.process_target = lambda x: x
            self.categorical_loss_func = self.hard_categorical
        else:
            self.process_target = lambda x: x.argmax(1)
            self.categorical_loss_func = AccuracyLoss(device=device)
        self.numerical_loss_func = self.c_l1_mean
        self.group_loss = BalancedErrorRateLoss(targetBer=ber_target, device=device)
        self.alpha_ = alpha_

    def hard_categorical(self, data, target):
        """
        As the model output tensor contains non zero elements (it is not always
        o_perfect =
        1, 0, 0
        0, 1, 0
        but instead (sometimes better values if a softmax is applied on cat output ?)
        o =
        0.46, 0.64, 0.001
        0.01, 0.02, 0.003

        We can not use argmax as it is not differentiable. We can not use equality input == target as we have decimals
        and we will most of the time end up with not even a single equality in the matrix, whereas doing
        input.argmax() == True might give some good results (remember that argmax is not differentiable).

        We can therefore do two solutions:
        
        hard = False: We pick the right element and we maximise the probability of the element, hence the accuracy
         overall: suppose target is
         t =
         1, 0, 0
         0, 0, 1
         We convert the target into indexes, yielding:
         0
         2
         And using the model output above, we obtain the vector
         v =
         0.46
         0.003
         which we compute the mean vm = v.mean() == accuracy, and we minimise the accuracy error 1 - vm

         hard = True, we want to impose some threshold on other values as well, not only on the specific target
         we compute the l1 per index, yielding
         0.54, 0.64, 0.001
         0.001, 0.002, 0.997
        We compute the mean per column
        cm =
         0.2705, 0.321, 0.499
         then we sum and divide by the number of element in cm different from 0. Yielding the proportion of mistakes if
         o were equal to o_perfect

        """
        all_wrong = self.c_l1_mean(data, target)
        denominator = (all_wrong != 0).sum()
        denominator = denominator if denominator != 0 else 1
        all_wrong = all_wrong.sum() / denominator
        return all_wrong

    def forward(self, data_, data_target, disc_pred=1, disc_target=1, group=1, *args, **kwargs):
        rec = self.numerical_loss_func(data_[:, self.num_indexes], data_target[:, self.num_indexes])
        rec = rec.view(-1)
        for c in self.cat_indexes:
            # if len(c) != 0:
                target = self.process_target(data_target[:, c])
                l = self.categorical_loss_func(data_[:, c], target)
                if len(rec) != 0:
                    rec = torch.cat((rec, l.view(-1)), 0)
                else:
                    rec = l.view(-1)

        # gr = self.group_loss(disc_pred, disc_target, group)
        return rec #(1 - self.alpha_) * rec, self.alpha_ * gr

    def __str__(self):
        return "{}(\n numerical Loss: {}\n Group Loss: {}\n Hard on Categorical: {}\n)" \
            .format(self.__class__.__name__, str(self.numerical_loss_func), str(self.group_loss), self.hard)
