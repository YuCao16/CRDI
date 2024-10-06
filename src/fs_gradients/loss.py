from typing import Any
import torch
import torch.nn.functional as F


def penalty_loss(gradients: Any, gradient_id: int) -> torch.Tensor:
    gradient = gradients.params[:, gradient_id]
    mean_gradient = torch.mean(gradient, dim=0)
    loss = F.mse_loss(
        gradient,
        mean_gradient.unsqueeze(0).expand_as(gradient),
        reduction="mean",
    )
    return loss
