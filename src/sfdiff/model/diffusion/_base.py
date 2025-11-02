# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
import math
import matplotlib.pyplot as plt


from sfdiff.utils import extract

class SFDiffBase(pl.LightningModule):
    def __init__(
        self,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        lr: float = 1e-3,
        dropout_rate: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dropout_rate = dropout_rate
        self.timesteps = timesteps
        
        self.betas = diffusion_scheduler(timesteps)
        self.sqrt_one_minus_beta = torch.sqrt(1.0 - self.betas)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        
        self.logs = {}

        self.context_length = context_length
        self.prediction_length = prediction_length
        
        self.losses_running_mean = torch.ones(timesteps, requires_grad=False)
        self.lr = lr
        self.best_crps = np.inf


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(1e12)
        )
        return [optimizer], {"scheduler": scheduler, "monitor": "train_loss"}

    def log(self, name, value, **kwargs):
        super().log(name, value, **kwargs)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if name not in self.logs:
            self.logs[name] = [value]
        else:
            self.logs[name].append(value)

    def get_logs(self):
        logs = self.logs
        logs["epochs"] = list(range(self.current_epoch))
        return pd.DataFrame.from_dict(logs)

    def q_sample(self, x_start, t, noise=None):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    '''
    @torch.no_grad()
    def p_sample(self,x_t,t):
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / self.alphas), t, x_t.shape)

        predicted_noise = self.backbone(x_t, t)

        
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean, None # No noise added at the last step

        
        posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t, device=self.device)
        sample = model_mean + torch.sqrt(posterior_variance_t) * noise
        return sample,None

    '''
    @torch.no_grad()
    def p_sample(self, x, t, t_index, y=None, h_fn=None, R_inv=None, base_strength=1.0, cheap=True, plot=False,
                 guidance=True, compute_snr=False, x_true=None):
        #given learnt score, predict unconditional model mean and then guide it using score of p(y_t|x_t) from tweedie approximation 
        betas_t = extract(self.betas, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        predicted_noise = self.backbone(x, t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        
        if guidance:
            assert h_fn is not None and R_inv is not None and y is not None, "h_fn and R_inv must be provided for guidance"
            if cheap:
                with torch.enable_grad():
                    grad_logp_y = self.observation_grad_cheap(x, t, y, h_fn, R_inv,self.context_length) 
            else:
                with torch.enable_grad():
                    grad_logp_y = self.observation_grad_expensive(x,t,y,h_fn,R_inv,self.context_length)
                    
            guide_strength = base_strength
            guided_mean = model_mean + guide_strength* betas_t * grad_logp_y
            
        
        else:
            guided_mean = model_mean



        if t_index == 0:
            sample = guided_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            sample = guided_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)

            
        snr_db = None
        if x_true is not None and compute_snr:
            if x_true.shape[0] == 1 and x.shape[0] > 1:
                x_true_trunc = x_true.expand(x.shape[0], -1, -1)
            else:
                x_true_trunc = x_true
            
            # compute SNR per sample
            snr_samples = []
            for s in range(sample.shape[0]):
                signal_power = torch.mean(x_true_trunc[s] ** 2)
                noise_power = torch.mean((sample[s] - x_true_trunc[s]) ** 2) + 1e-9
                snr_samples.append(10 * torch.log10(signal_power / noise_power))
            
            # mean over samples
            snr_db = torch.stack(snr_samples).mean()
            
            
        if plot:
            os.makedirs("diffusion_frames", exist_ok=True)
            obs_dim = guided_mean.shape[2]
            
            if obs_dim == 3:
                # 3D plot
                fig = plt.figure(figsize=(6, 5))
                ax_3d = fig.add_subplot(111, projection='3d')
                
                guided_cpu = guided_mean[0].detach().cpu().numpy()  # shape: [seq_len, 3]
                ax_3d.plot(guided_cpu[:, 0], guided_cpu[:, 1], guided_cpu[:, 2], 
                        color='black', linewidth=2, label=f"Predicted (t={t_index})")
                
                if guidance:
                    assert y is not None, "y must be provided for plotting observations"
                    y_cpu = y[0].detach().cpu().numpy()
                    ax_3d.scatter(y_cpu[:, 0], y_cpu[:, 1], y_cpu[:, 2], 
                                color='orange', label="Observations", alpha=0.7)
                
                ax_3d.set_xlabel("X")
                ax_3d.set_ylabel("Y")
                ax_3d.set_zlabel("Z")
                ax_3d.set_title(f"Reverse Diffusion Step {t_index}")
                ax_3d.legend()
                plt.tight_layout()
                plt.savefig(f"diffusion_frames/step_{t_index:04d}.png")
                plt.close()
                
            else:
                # 1D plot
                plt.figure(figsize=(6, 3))
                guided_cpu = guided_mean[0, :, 0].detach().cpu().numpy()  # shape: [seq_len]
                plt.plot(guided_cpu, label=f"Predicted (t={t_index})", alpha=0.8)
                
                if guidance:
                    assert y is not None, "y must be provided for plotting observations"
                    y_cpu = y[0, :, 0].detach().cpu().numpy()
                    plt.ylim(min(y_cpu), max(y_cpu))
                    plt.plot(y_cpu, label="Observations", alpha=0.7)
                
                plt.title(f"Reverse Diffusion Step {t_index}")
                plt.legend(loc='lower left')
                plt.tight_layout()
                plt.savefig(f"diffusion_frames/step_{t_index:04d}.png")
                plt.close()
        
        return sample, snr_db

        
    def observation_grad_cheap(self, x_t, t, y, h_fn, R_inv,context_len):
        # no autograd through backbone required (only need grad through h)
        x_t = x_t.requires_grad_(True)
        eps = self.backbone(x_t, t)
        sqrt_bar_alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_bar = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        x0 = (x_t - sqrt_one_minus_bar * eps) / sqrt_bar_alpha
        y_pred = h_fn(x0)
        
        # only guide on known context
        resid = y[:, :context_len, :] - y_pred[:, :context_len, :]
        r = R_inv(resid)

        # compute J_h^T r per-batch (cheap)
        Jt_r = []
        for i in range(x0.shape[0]):
            scalar = (y_pred[i, :context_len, :].reshape(-1) * r[i].reshape(-1)).sum()
            gi = torch.autograd.grad(scalar, x0, retain_graph=True, create_graph=False)[0][i]
            Jt_r.append(gi)
        Jt_r = torch.stack(Jt_r, dim=0)

        return Jt_r

    def observation_grad_expensive(self, x_t, t, y, h_fn, R_inv, context_len):
        x_t = x_t.requires_grad_(True)
        eps = self.backbone(x_t, t)
        
        sqrt_bar_alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_bar = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        x0 = (x_t - sqrt_one_minus_bar * eps) / sqrt_bar_alpha
        y_pred = h_fn(x0)
        
        # only guide on known context
        resid = y[:, :context_len, :] - y_pred[:, :context_len, :]
        r = R_inv(resid)  # [B, context_len, ydim]

        w = []
        for i in range(x0.shape[0]):
            scalar = (y_pred[i, :context_len, :].reshape(-1) * r[i].reshape(-1)).sum()
            wi = torch.autograd.grad(scalar, x0, retain_graph=True, create_graph=True)[0][i]
            w.append(wi)
        w = torch.stack(w, dim=0)  # [B, state_dim]

        v = []
        for i in range(x_t.shape[0]):
            # restrict vjp through eps to context_len as well
            scalar2 = (eps[i, :context_len, :] * w[i, :context_len, :]).sum()
            vi = torch.autograd.grad(scalar2, x_t, retain_graph=True, create_graph=False)[0][i]
            v.append(vi)
        v = torch.stack(v, dim=0)

        obs_grad_wrt_xt = (w - sqrt_one_minus_bar * v) / sqrt_bar_alpha
        return obs_grad_wrt_xt


    @torch.no_grad()
    def sample(self, noise, y, h_fn, R_inv):
        device = next(self.backbone.parameters()).device
        batch_size, seq_len, ch = noise.shape
        context_len = self.context_length

        seq = noise
        seqs = [seq.cpu()]

        for i in reversed(range(0, self.timesteps)):
            seq = self.p_sample(
                seq,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                i,
                y,
                h_fn,
                R_inv
            )
            seqs.append(seq.cpu().numpy())

        return np.stack(seqs, axis=0)

    def fast_denoise(self, xt, t):
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, xt.shape)
        score = self.backbone(xt, t)
        return (xt - sqrt_one_minus_alphas_cumprod_t * score) / sqrt_alphas_cumprod_t

    def fast_sample(self,y,num_steps=None):
        device = next(self.backbone.parameters()).device
        batch_size, seq_len, ch = y.shape

        if num_steps is None:
            num_steps = self.timesteps
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device)
        else:
            # Linear DDIM sampling schedule
            timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()

        # Initialize with standard normal noise
        x = torch.randn(batch_size, seq_len, ch, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            # fast deterministic denoising
            x = self.fast_denoise(x, t_tensor)
        
        return x

    def forward(self, x, mask):
        raise NotImplementedError()

    def training_step(self, data, idx):
        device = next(self.backbone.parameters()).device


        x_start = torch.as_tensor(
            torch.cat([data["past_observation"], data["future_observation"]], dim=1),
            dtype=torch.float32,
            device=device
        )

        t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=device).long()

        # add noise and compute denoising loss
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)  # forward diffusion
        predicted_noise = self.backbone(x_t, t) # unconditional model

        loss = F.mse_loss(predicted_noise, noise)

        self.log("elbo_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss, "elbo_loss": loss}


    def training_epoch_end(self, outputs):
        epoch_loss = sum(x["loss"] for x in outputs) / len(outputs)
        elbo_loss = sum(x["elbo_loss"] for x in outputs) / len(outputs)
        self.log("train_loss", epoch_loss)
        self.log("train_elbo_loss", elbo_loss)


