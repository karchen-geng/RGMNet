import torch
import torch.nn.functional as F


# Soft aggregation from STM
def aggregate(prob, keep_bg=False):
    k = prob.shape
    new_prob = torch.cat([
        torch.prod(1-prob, dim=0, keepdim=True),
        prob
    ], 0).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))  # 3,1,h,w

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]