import argparse
import os
import sys
from functools import partial
from tqdm import tqdm
from typing import Any, Callable, List, Tuple, Optional, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader
from torchvision import transforms

from lightning.fabric import Fabric
import lightning.fabric.strategies.ddp as ddp

from src.utils import print_config_tree, modify_list_arguments

from src.fs_gradients.utils import (
    update_args_by_category,
    split_list,
    generate_random_t,
)
from src.fs_gradients.loss import penalty_loss
from src.fs_gradients.utils import load_config, save_gradients
from src.fs_gradients.model import GradientConfig
from src.fs_gradients.diffusion import q_sample_noise
from src.fs_gradients.dataset import FewShotDataset

from guided_diffusion.guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ffhq_train.yaml",
        help="Path to the YAML config file",
    )
    args, _ = parser.parse_known_args()
    defaults = load_config(args.config)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    modify_list_arguments(parser)
    return parser


def postprocess_args(args: argparse.Namespace) -> None:
    update_args_by_category(args)

    if args.num_gradient > (args.t_end - args.t_start):
        args.num_gradient = args.t_end - args.t_start
    args.experiment_gradient_path = f"checkpoints"
    os.makedirs(args.experiment_gradient_path, exist_ok=True)


def get_target_dataloader(
    args: argparse.Namespace,
    sampler: Optional[RandomSampler] = None,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    dataset = FewShotDataset(csv_file=args.csv_file, transform=transform)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler,
    )


def get_model_and_diffusion(args: argparse.Namespace) -> Tuple:
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path))
    assert model.num_classes is None
    model.eval()
    return model, diffusion


def get_optimizer(
    args: argparse.Namespace, gradients: GradientConfig
) -> torch.optim.Optimizer:
    params = [gradients._params]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    return optimizer


def get_gradients(args: argparse.Namespace) -> GradientConfig:
    return GradientConfig(
        num_images=args.num_samples,
        num_gradient=args.num_gradient,
    )


@torch.no_grad()
def construct_x_t(
    args: argparse.Namespace,
    img: torch.Tensor,
    diffusion: SpacedDiffusion,
    device: Union[str, torch.device] = "cuda",
    t_des: Optional[Union[int, float]] = None,
) -> torch.Tensor:
    t = torch.tensor(t_des, device=device).repeat(img.shape[0])
    return diffusion.q_sample(
        img, t, noise=q_sample_noise(img, args.random_q_noise)
    )


def train_method(
    args: argparse.Namespace,
    model: nn.Module,
    diffusion: SpacedDiffusion,
    gradients: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cond_fn: Callable,
    prefix_name: str,
) -> None:
    model_kwargs = {}
    outputs = None
    timestep_splits = split_list(args.t_start, args.t_end, args.num_gradient)
    pbar = tqdm(
        list(reversed(timestep_splits)),
        disable=(len(timestep_splits) == 1) or not args.tqdm,
    )
    for gradient_id, timestep_split in enumerate(pbar):
        gradient_id = len(timestep_splits) - gradient_id - 1
        pbar.set_description(f"Gradient ID {gradient_id}")
        pbar_in = tqdm(
            range(args.epochs),
            leave=(len(timestep_splits) == 1),
            disable=not args.tqdm,
        )
        for epoch in pbar_in:
            running_loss_total = 0.0
            timestep_list = []
            for batch in data_loader:
                timestep_start, timestep_end = timestep_split
                num_element = max(int(0.8 * (timestep_end - timestep_start)), 1)
                timestep_list = generate_random_t(
                    timestep_start,
                    timestep_end,
                    num_element,
                    fabric.device,
                )
                running_loss, outputs = train_step(
                    args,
                    model,
                    diffusion,
                    optimizer,
                    batch,
                    timestep_list,
                    cond_fn,
                    model_kwargs,
                    gradient_id,
                    gradients=gradients,
                )
                running_loss_total += running_loss
            epoch_loss = running_loss_total / sum(
                t.numel() for t in timestep_list
            )
            pbar_in.set_description(
                f"Epoch {epoch + 1}/{args.epochs} Loss: {epoch_loss:.4f}"
            )
        assert outputs is not None
    if args.save_gradients:
        save_gradients(args, gradients, prefix_name)


def train_step(
    args: argparse.Namespace,
    model: nn.Module,
    diffusion: SpacedDiffusion,
    optimizer: torch.optim.Optimizer,
    batch: List,
    timestep_list: torch.Tensor,
    cond_fn: Callable,
    model_kwargs: dict,
    gradient_id: int,
    gradients: Optional[nn.Module] = None,
) -> Tuple[Union[int, float], Any]:
    running_loss, loss = 0.0, 0.0
    outputs = None
    x_0, y = batch
    if args.normalization:
        x_0 = x_0 * 2 - 1
    optimizer.zero_grad()

    pbar = tqdm(
        reversed(timestep_list),
        leave=False,
        disable=(len(timestep_list) == 1) or not args.tqdm,
        desc="Steps         ",
    )
    cond_fn_label = partial(cond_fn, label=y, gradient_id=gradient_id)
    for t in pbar:
        pbar.set_description(f"Steps {t.item():02d}      ")
        t = t.unsqueeze(dim=0).repeat(args.batch_size)
        assert t[0].item() < (args.t_end)
        x_t = construct_x_t(
            args,
            x_0,
            diffusion,
            device=fabric.device,
            t_des=t[0].item(),
        )
        assert x_t.shape == x_0.shape
        if t[0].item() > 0:
            y_t_prev = diffusion.q_sample(
                x_0, t - 1, noise=q_sample_noise(x_0, args.random_q_noise)
            )
        else:
            y_t_prev = x_0
        outputs = diffusion.ddim_sample(
            model,
            x_t,
            t,
            cond_fn=cond_fn_label,
            model_kwargs=model_kwargs,
        )
        assert outputs["sample"].shape == y_t_prev.shape
        loss = nn.functional.mse_loss(
            x_0, outputs["pred_xstart"]
        ) + nn.functional.mse_loss(y_t_prev, outputs["sample"])
        loss = loss + args.penalty_weight * penalty_loss(gradients, gradient_id)
        fabric.backward(loss)
        running_loss += loss.item()
    optimizer.step()
    return running_loss, outputs


def main() -> None:
    args = create_argparser().parse_args()
    postprocess_args(args)
    print_config_tree(vars(args)) if args.print_config else None

    model, diffusion = get_model_and_diffusion(args)
    gradients = get_gradients(args)
    data_loader = get_target_dataloader(args)
    optimizer = get_optimizer(args, gradients)

    model = fabric.setup(model)
    gradients, optimizer = fabric.setup(gradients, optimizer)
    data_loader = fabric.setup_dataloaders(data_loader)

    def cond_fn(
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        label: Optional[Union[torch.Tensor, int]] = None,
        gradient_id: int = 0,
    ) -> torch.Tensor:
        return gradients(label, gradient_id) * args.classifier_scale

    train_method(
        args,
        model,
        diffusion,
        gradients,
        data_loader,
        optimizer,
        cond_fn,
        args.category,
    )


if __name__ == "__main__":
    fabric = Fabric(
        accelerator="gpu",
        strategy=ddp.DDPStrategy(find_unused_parameters=False),
    )
    fabric.launch()
    fabric.seed_everything(2024)

    main()
