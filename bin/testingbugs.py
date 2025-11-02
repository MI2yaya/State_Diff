#!/usr/bin/env python3
"""
TSDiff Continual Learning Comparison Plots (State Forecast)
"""

import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sfdiff.configs as diffusion_configs

from sfdiff.model.diffusion.diff import SFDiff
from sfdiff.dataset import get_custom_dataset
from sfdiff.utils import extract


    
def load_info(config):
    device = config['device']
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)

    scaling = int(config['dt']**-1)
    context_length = config["context_length"] * scaling
    prediction_length = config["prediction_length"] * scaling

    dataset, generator = get_custom_dataset(config["dataset"],
        samples=0,
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        dt=config['dt'],
        q=config['q'],
        r=config['r'],
        observation_dim=config['observation_dim'],
    )
    h_fn = generator.h_fn
    R_inv = generator.R_inv

    model = SFDiff(
        **getattr(diffusion_configs, config["diffusion_config"]),
        observation_dim=config["observation_dim"],
        context_length=context_length,
        prediction_length=prediction_length,
        lr=config["lr"],
        init_skip=config["init_skip"],
        h_fn=h_fn,
        R_inv=R_inv,
        modelType=config['diffusion_config'].split('_')[1],
    )

    # Load state_dict
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model

   

def test(config):
    model = load_info(config)
    model.eval()
    device = next(model.backbone.parameters()).device

    B = 2
    L = model.context_length + model.prediction_length
    C = model.observation_dim

    x0 = torch.randn(B, L, C, device=device)      # clean signals
    t = torch.full((B,), model.timesteps - 1, device=device, dtype=torch.long)

    eps_true = torch.randn_like(x0, device=device)
    x_T = model.q_sample(x0, t, eps_true)         # produce x_T using eps_true

    # build model_mean if backbone predicted eps_true perfectly
    betas_t = extract(model.betas, t, x_T.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(model.sqrt_one_minus_alphas_cumprod, t, x_T.shape)
    sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / model.alphas), t, x_T.shape)

    model_mean_true = sqrt_recip_alphas_t * (x_T - betas_t * eps_true / (sqrt_one_minus_alphas_cumprod_t + 1e-12))

    print("x0 stats:", x0.mean().item(), x0.std().item())
    print("x_T stats:", x_T.mean().item(), x_T.std().item())
    print("model_mean_true stats:", model_mean_true.mean().item(), model_mean_true.std().item())

    # Optional: deterministic backward step to approximate x_{T-1}
    x_prev_deterministic = model_mean_true
    print("x_prev_deterministic stats:", x_prev_deterministic.mean().item(), x_prev_deterministic.std().item())
    
    B = 8
    x0_batch = torch.randn(B, L, C, device=device)
    t = torch.randint(0, model.timesteps, (B,), device=device)
    eps = torch.randn_like(x0_batch, device=device)
    x_t = model.q_sample(x0_batch, t, eps)

    with torch.no_grad():
        pred = model.backbone(x_t, t)  # predicted_noise

    mse_per_sample = ((pred - eps)**2).mean(dim=[1,2]).cpu().numpy()
    print("mse per sample:", mse_per_sample)

    # overall stats & correlation
    pred_flat = pred.detach().cpu().numpy().ravel()
    eps_flat = eps.detach().cpu().numpy().ravel()
    import numpy as np
    corr = np.corrcoef(pred_flat, eps_flat)[0,1]
    print("global mse:", ((pred-eps)**2).mean().item(), "corr:", corr)

    # per-timestep bucketed mse
    import collections
    by_t = collections.defaultdict(list)
    for i,tt in enumerate(t.cpu().numpy()):
        by_t[tt].append(((pred[i]-eps[i])**2).mean().item())
    for tt in sorted(by_t)[:10]:
        print(f"t={tt} mean mse={np.mean(by_t[tt]):.6f} count={len(by_t[tt])}")



def main(config_path):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    test(config)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    args = parser.parse_args()

    main(args.config)
