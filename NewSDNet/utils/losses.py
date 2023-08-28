import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_penalty(x, epsilon_squared=0.01):
    charbonnier_loss = torch.sqrt(x * x + epsilon_squared)
    return charbonnier_loss


def KL_divergence(logvar, mu):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()


def dice_loss(pred, target):
    smooth = 0.1

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    loss = ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean()

    return 1 - loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            print(f"\nSHAPE DI INPUT 1: {input.shape}")
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            print(f"\nSHAPE DI INPUT 2: {input.shape}")
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            print(f"\nSHAPE DI INPUT 3: {input.shape}")
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
            print(f"\nSHAPE DI INPUT 4: {input.shape}")
        print(f"\nSHAPE TARGET 1: {target.shape}")
        target = target.view(-1, 1).long()
        print(f"\nSHAPE TARGET 2: {target.shape}")

        logpt = F.log_softmax(input)
        print(f"\nSHAPE LOGPT 1: {logpt.shape}")
        logpt = logpt.gather(1, target)
        print(f"\nSHAPE LOGPT 2: {logpt.shape}")
        logpt = logpt.view(-1)
        print(f"\nSHAPE LOGPT 3: {logpt.shape}")
        pt = logpt.data.exp()
        print(f"\nSHAPE PT: {pt.shape}")

        loss = -1 * (1 - pt) ** self.gamma * logpt
        print(f"\nLOSS SHAPE: {loss.shape}")
        if self.size_average:
            print(f"\nLOSS AFTER mean: {loss.mean().shape}")
            return loss.mean()

        else:
            return loss.sum()


def HSIC_lossfunc(x, y):
    assert x.dim() == y.dim() == 2
    assert x.size(0) == y.size(0)
    m = x.size(0)
    h = torch.eye(m) - 1 / m
    h = h.to(x.device)
    K_x = gaussian_kernel(x)
    K_y = gaussian_kernel(y)
    return K_x.mm(h).mm(K_y).trace() / (m - 1 + 1e-10)


def gaussian_kernel(x, y=None, sigma=5):
    if y is None:
        y = x
    assert x.dim() == y.dim() == 2
    assert x.size() == y.size()
    z = ((x.unsqueeze(0) - y.unsqueeze(1)) ** 2).sum(-1)
    return torch.exp(-0.5 * z / (sigma * sigma))


class GeneralizedCELoss(nn.Module):
    def __init__(self, q: float):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        if logits.dim() > 2:
            logits_for_weight = logits.view(
                logits.size(0), logits.size(1), -1
            )  # N,C,H,W => N,C,H*W
            logits_for_weight = logits_for_weight.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logits_for_weight = logits_for_weight.contiguous().view(
                -1, logits_for_weight.size(2)
            )  # N,H*W,C => N*H*W,C
        targets_for_weight = targets.view(-1, 1).long()

        p = F.log_softmax(logits_for_weight)
        assert p.mean().item() is not None
        Yg = torch.gather(p, 1, targets_for_weight)  # not sure this unsqueeze is needed

        # modify gradient of cross entropy
        loss_weight = Yg.view(-1).detach()
        # loss_weight = (
        #     Yg.squeeze().detach() ** self.q
        # )  # Do we really need *self.q? I think like now is correct.
        assert Yg.mean().item() is not None

        # note that we don't return the average but the loss for each datum separately
        loss = F.cross_entropy(logits, targets, reduction="mean") * loss_weight
        print(f"\nSHAPE DI LOSS GCE: {loss.shape}")
        print(f"\nSHAPE DI LOSS GCE after mean: {loss.mean().shape}")

        return loss.mean()
