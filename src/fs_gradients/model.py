import torch
from torch import nn


class GradientConfig(nn.Module):
    def __init__(
        self,
        num_images: int,
        num_gradient: int = 1,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.num_gradient = num_gradient

        shape = (num_images, num_gradient, 3, image_size, image_size)

        self._params = nn.Parameter(torch.randn(shape), requires_grad=True)

    @property
    def params(self) -> torch.Tensor:
        return self._params

    def forward(
        self,
        idx: torch.Tensor,
        gradient_id: int = 1,
        mode: str = "train",
    ) -> torch.Tensor:
        if mode == "sample":
            batch_size = idx.size(0)
            idx = torch.randperm(self._params.size(0))[:batch_size].to(
                idx.device
            )
        return self._params[idx, gradient_id]
