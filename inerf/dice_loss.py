import torch
import torch.cuda.amp as amp
import torch.nn as nn


##
# v2: self-derived grad formula
class SoftDiceLossV2(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV2, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        logits = logits.view(1, -1)
        labels = labels.view(1, -1)
        loss = SoftDiceLossV2Func.apply(logits, labels, self.p, self.smooth)
        return loss


class SoftDiceLossV2Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, p, smooth):
        '''
        inputs:
            logits: (N, L)
            labels: (N, L)
        outpus:
            loss: (N,)
        '''
        #  logits = logits.float()

        probs = torch.sigmoid(logits)
        numer = 2 * (probs * labels).sum(dim=1) + smooth
        denor = (probs.pow(p) + labels.pow(p)).sum(dim=1) + smooth
        loss = 1. - numer / denor

        ctx.vars = probs, labels, numer, denor, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of soft-dice loss
        '''
        probs, labels, numer, denor, p, smooth = ctx.vars

        numer, denor = numer.view(-1, 1), denor.view(-1, 1)

        term1 = (1. - probs).mul_(2).mul_(labels).mul_(probs).div_(denor)

        term2 = probs.pow(p).mul_(1. - probs).mul_(numer).mul_(p).div_(denor.pow_(2))

        grads = term2.sub_(term1).mul_(grad_output)

        return grads, None, None, None
