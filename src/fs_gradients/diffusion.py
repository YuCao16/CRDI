import torch


def q_sample_noise(x, random=False, device="cuda"):
    x_size = x.size()
    if random:
        return torch.randn_like(x)
    return torch.randn(
        size=x_size,
        generator=torch.Generator(device=device).manual_seed(0),
        device=device,
    )


def ddim_sample_loop_progressive(
    self,
    model,
    shape,
    args,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    eta=0.0,
):
    """
    Use DDIM to sample from the model and yield intermediate samples from
    each timestep of DDIM.

    Same usage as p_sample_loop_progressive().
    """
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(shape, (tuple, list))
    if noise is not None:
        img = noise.to(device)
        indices = list(range(args.t_end))[::-1]
    else:
        img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

    if progress:
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm

        indices = tqdm(indices, desc="Timestep(ddim)")

    for i in indices:
        t = torch.tensor([i] * shape[0], device=device)
        _cond_fn = (
            cond_fn if (int(args.t_start) <= i < int(args.t_end)) else None
        )
        with torch.no_grad():
            out = self.ddim_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=_cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            yield out
            img = out["sample"]
