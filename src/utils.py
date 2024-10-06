import argparse
import os
import PIL
from PIL import Image
from typing import Any
import numpy as np
import torch as th
from rich.tree import Tree
from rich import print as rprint


def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed[0])
    try:
        from IPython.display import display  # type: ignore

        display(f"Image at step {i}")
        display(image_pil)
    except ImportError:
        print(f"Image at step {i}")
        image_pil.show()


def tensor2img(sample):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed[0])
    return image_pil


def img2tensor(img_path):
    img = Image.open(img_path)
    img = img.resize((256, 256))

    img_np = np.array(img)

    img_np = (img_np - 127.5) / 127.5

    img_tensor = th.tensor(img_np).float()
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    return img_tensor


def create_grid(images_list, rows, cols):
    assert (
        len(images_list) == rows * cols
    ), "Number of images does not match grid size"

    w, h = images_list[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, image in enumerate(images_list):
        grid.paste(image, box=(i % cols * w, i // cols * h))

    return grid


def find_nearest_index(lst, a):
    """
    Find the index of the element in the list that is closest to the given value 'a'.
    If 'a' is in the list, return its index. Otherwise, return the index of the nearest element.
    """
    if a in lst:
        return a, lst.index(a)

    # Find the nearest value
    nearest_value = min(lst, key=lambda x: abs(x - a))
    print(f"timestep has been changed to: {nearest_value}")
    return nearest_value, lst.index(nearest_value)


def print0(*args: Any, **kwargs: Any) -> None:
    if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
        print(*args, **kwargs)


def print_config_tree(cfg: dict, parent=None):
    # Create a new tree if this is the top-level call
    if parent is None:
        parent = Tree("CONFIG")

    for key, value in cfg.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recursively handle it
            branch = parent.add(f"[bold]{key}")
            print_config_tree(value, branch)
        else:
            # If the value is not a dictionary, add it directly to the parent branch
            parent.add(f"{key}: {value}")

    # Print the entire tree if this is the top-level call
    if parent.label == "CONFIG":
        rprint(parent)


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        else:
            return False  # Terminal running IPython
    except NameError:
        return False  # Probably standard Python interpreter


def modify_list_arguments(parser):
    for action in parser._actions:
            if isinstance(action.default, list):
                def type_func(v):
                    if v is None:
                        return action.default
                    else:
                        return v.split(',')
                action.nargs = '?'
                action.const = action.default
                action.type = type_func
