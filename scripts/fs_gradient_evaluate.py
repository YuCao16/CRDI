import argparse
from argparse import Namespace
import os
import sys
from typing import Callable, List, Optional, Tuple, Union
from functools import partial
from types import MethodType
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


from src.utils import print_config_tree

from src.fs_gradients.model import GradientConfig
from src.fs_gradients.dataset import FewShotDataset
from src.fs_gradients.evaluation import Evaluator
from src.fs_gradients.diffusion import (
    ddim_sample_loop_progressive,
    q_sample_noise,
)
from src.fs_gradients.utils import (
    update_args_by_category,
    get_timestep_dict,
    load_config,
)

from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def deterministic(seed: Optional[int]) -> None:
    if seed is None:
        seed = 2024
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ffhq_generate.yaml",
        help="Path to the YAML config file",
    )
    args, _ = parser.parse_known_args()
    defaults = load_config(args.config)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def postprocess_args(args: Namespace) -> None:
    update_args_by_category(args)
    assert os.path.exists(args.experiment_gradient_path)


def get_dataloader(args: Namespace) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    dataset = FewShotDataset(csv_file=args.csv_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return data_loader


def get_model_and_diffusion(
    args: Namespace, device: Union[str, torch.device]
) -> Tuple:
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path))
    assert model.num_classes is None
    model.to(device)
    model.eval()
    ddim_sample_loop_progressive_args = partial(
        ddim_sample_loop_progressive, args=args
    )
    diffusion.ddim_sample_loop_progressive = MethodType(
        ddim_sample_loop_progressive_args, diffusion
    )
    return model, diffusion


def get_gradients(args: Namespace) -> GradientConfig:
    gradients = GradientConfig(
        num_images=args.num_samples,
        num_gradient=args.num_gradient,
    )
    state_dict = torch.load(args.experiment_gradient_path)
    gradients.load_state_dict(state_dict)
    return gradients


def generate_samples(
    args: Namespace,
    data_loader: DataLoader,
    cond_fn: Callable,
    model_fn: Callable,
    sample_fn: Callable,
    q_sample: Callable,
    device: Union[str, torch.device],
) -> List:
    all_images = []
    pbar = tqdm(
        total=args.num_evaluate,
        desc="Generating Samples",
        disable=not args.tqdm,
    )
    while len(all_images) < args.num_evaluate:
        for batch in data_loader:
            x_0, y = batch
            if args.normalization:
                x_0 = x_0 * 2 - 1
            timestep = torch.tensor(args.t_end - 1).to(x_0.device)
            x_t = q_sample(
                x_0.cuda(),
                timestep.cuda(),
                noise=q_sample_noise(x_0.cuda(), args.random_q_noise),
            )
            model_kwargs = {}
            cond_fn_label = partial(cond_fn, label=y)
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_label,
                device=device,
                progress=False,
                noise=x_t,
            )
            if args.normalization:
                sample = (sample + 1) / 2
            sample = sample.contiguous()
            all_images.extend([[s.cpu().numpy()] for s in sample])
        pbar.update(args.batch_size)

    pbar.close()
    return all_images


def main() -> None:
    args = create_argparser().parse_args()
    postprocess_args(args)
    print_config_tree(vars(args)) if args.print_config else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, diffusion = get_model_and_diffusion(args, device)

    print(f"using checkpoint: {args.experiment_gradient_path}")
    gradients = get_gradients(args).to(device)
    data_loader = get_dataloader(args)
    timestep_dict = get_timestep_dict(
        args.t_start,
        args.t_end,
        args.num_gradient,
        timestep_map=diffusion.timestep_map,
    )

    @torch.no_grad()
    def cond_fn(x, t, y=None, label=None):
        assert timestep_dict is not None

        gradient_id = timestep_dict[t[0].item()]
        _gradients = (
            gradients(label, gradient_id=gradient_id, mode="sample")
            * args.classifier_scale
        )
        if args.anneal_ptb:
            anneal_scale = (
                (args.t_start - diffusion.timestep_map.index(t[0].item()))
                / (args.t_start - args.t_end)
                * args.anneal_scale
            )
            _gradients = _gradients + anneal_scale * torch.randn_like(
                _gradients
            )
        return _gradients

    def model_fn(x, t, y=None):
        return model(x, t, y if args.class_cond else None)

    sample_fn = diffusion.ddim_sample_loop
    q_sample = diffusion.q_sample

    all_images = generate_samples(
        args, data_loader, cond_fn, model_fn, sample_fn, q_sample, device
    )
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_evaluate]
    np.save("arr.npy", arr)

    evaluator = Evaluator(
        args,
        torch.from_numpy(arr),
        f"/path/to/project/datasets/fid_npz/{args.category}.npz",
        args.lpips_cluster_size,
    )

    fid_score = evaluator.calc_fid()
    print(f"FID: {fid_score}")
    intra_lpips = evaluator.calc_intra_lpips()
    print("Intra-LPIPS: ", intra_lpips)


if __name__ == "__main__":
    # deterministic(2024)
    main()
