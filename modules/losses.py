import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Binary FocalLoss, a CrossEntropy variant based on two logits.
    - Input: logits [batch, 2], targets [batch] (0/1).
    - Calculation: take the positive-class probability after softmax, use BCE form, and scale by (1-pt)^gamma.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs_pos = torch.softmax(logits, dim=1)[:, 1]
        targets = targets.float()
        pt = probs_pos * targets + (1.0 - probs_pos) * (1.0 - targets)
        ce = nn.functional.binary_cross_entropy(probs_pos, targets, reduction='none')
        loss = self.alpha * (1.0 - pt).pow(self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


