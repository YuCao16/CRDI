from argparse import Namespace
import os
from typing import Any, List, Optional, Union
import yaml

import torch


CATEGORIES_TO_CSV_FILES = {
    "babies": "datasets/babies_target/babies.csv",
    "babiesone": "datasets/babiesone_target/babiesone.csv",
    "babiesfive": "datasets/babiesfive_target/babiesfive.csv",
    "metfaces": "datasets/metfaces_target/metfaces.csv",
    "metfacesone": "datasets/metfacesone_target/metfacesone.csv",
    "metfacesfive": "datasets/metfacesfive_target/metfacesfive.csv",
    "sunglasses": "datasets/sunglasses_target/sunglasses.csv",
    "sunglassesone": "datasets/sunglassesone_target/sunglassesone.csv",
    "sunglassesfive": "datasets/sunglassesfive_target/sunglassesfive.csv",
}


def generate_random_t(
    min_val: int = 0,
    max_val: int = 25,
    num_element: int = 10,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    range_size = (max_val - min_val) / num_element
    starts = torch.arange(min_val, max_val, range_size, device=device)
    random_offsets = torch.rand(num_element, device=device) * range_size
    random_ints = (starts + random_offsets).int()
    return random_ints


def split_list(start: int, end: int, num_step: int) -> List:
    total_length = end - start

    quotient, remainder = divmod(total_length, num_step)
    start_list = []
    current_start = start
    for i in range(num_step):
        start_list.append(current_start)
        current_start += quotient + 1 if i < remainder else quotient
    end_list = start_list[1:] + [end]
    return list(zip(start_list, end_list))


def get_timestep_lists(
    start: int,
    end: int,
    num_segments: int,
    device: Union[str, torch.device] = "cpu",
) -> List:
    segments = split_list(start, end, num_segments)
    max_segment_length = max(end - start for start, end in segments)
    accumulate_step = max(1, int(max_segment_length * 0.8))
    timestep_lists = [
        generate_random_t(segment[0], segment[1], accumulate_step, device)
        for segment in segments
    ]
    return timestep_lists


def update_args_by_category(args: Namespace) -> None:
    if args.category not in CATEGORIES_TO_CSV_FILES:
        raise NotImplementedError
    args.csv_file = CATEGORIES_TO_CSV_FILES[args.category]


def get_timestep_dict(
    t_start: int,
    t_end: int,
    num_gradient: int,
    timestep_map: Optional[List] = None,
) -> dict:
    timestep_segment = split_list(t_start, t_end, num_gradient)
    timestep_dict = dict()
    for idx, segment in enumerate(timestep_segment):
        for j in reversed(range(*segment)):
            if timestep_map is not None:
                timestep_dict[timestep_map[j]] = idx
            else:
                timestep_dict[j] = idx
    return timestep_dict


def load_config(config_file: str) -> Any:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_gradients(args: Namespace, gradients: Any, prefix_name: str) -> None:
    torch.save(
        gradients.state_dict(),
        os.path.join(args.experiment_gradient_path, f"model_{prefix_name}.pth"),
    )
    torch.save(
        gradients.params.detach().cpu(),
        os.path.join(
            args.experiment_gradient_path, f"params_{prefix_name}.pth"
        ),
    )
