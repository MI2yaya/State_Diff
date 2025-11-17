# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import torch
from sfdiff.arch.backbones import SequenceBackbone, UNetBackbone
from sfdiff.model.diffusion._base import SFDiffBase
from sfdiff.utils import make_diffusion_gif


class SFDiff(SFDiffBase):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        observation_dim,
        h_fn,
        R_inv,
        init_skip=True,
        lr=1e-3,
        dropout_rate=0.01,
        modelType='s4'
    ):
        super().__init__(
            timesteps=timesteps,
            diffusion_scheduler=diffusion_scheduler,
            context_length=context_length,
            prediction_length=prediction_length,
            lr=lr,
            dropout_rate=dropout_rate,
        )
        self.lags_seq=[0] #just for callback past_length=self.context_length + max(self.model.lags_seq),
        backbone_parameters["observation_dim"] = observation_dim
        backbone_parameters['output_dim']=observation_dim
        print(backbone_parameters)

        self.modelType=modelType
        
        if modelType=='s4' or modelType=="s5":
            print(f'Running {modelType}')
            backbone_parameters["dropout"] = dropout_rate
            self.backbone = SequenceBackbone(
                **backbone_parameters,
                init_skip=init_skip,
            )
            
        elif modelType=='unet':
            print('Running unet')
            self.backbone = UNetBackbone(
                **backbone_parameters
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_parameters['type']}")
        
        self.ema_rate = []  # [0.9999]
        self.ema_state_dicts = [
            copy.deepcopy(self.backbone.state_dict())
            for _ in range(len(self.ema_rate))
        ]
        self.observation_dim = observation_dim
        self.h_fn = h_fn
        self.R_inv = R_inv




    @torch.no_grad()
    def sample_n(
        self,
        y,
        num_samples: int = 1,
        cheap=True,
        base_strength=0.1,
        plot=False,
        guidance=True,
        
    ):
        device = next(self.backbone.parameters()).device
        context_len = self.context_length
        full_len = context_len + self.prediction_length
        seq_len = full_len
        known_len = y.shape[1]

        # initial noise
        samples = torch.randn((num_samples, seq_len, self.observation_dim), device=device)


        if y is not None:
            mask = torch.zeros((num_samples, seq_len, self.observation_dim), device=device)
            mask[:, :known_len, :] = 1
            known_full = torch.zeros((num_samples, seq_len, self.observation_dim), device=device)
            known_full[:, :known_len, :] = y
        else:
            mask = None
            known_full = None


        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            x_prev_uncond = self.p_sample(
                x=samples,
                t=t,
                t_index=i,
                y=y,
                h_fn=self.h_fn,
                R_inv=self.R_inv,
                base_strength=base_strength,
                cheap=cheap,
                plot=plot,
                guidance=guidance
            )

            #x_prev_uncond, snr = self.p_sample(x_t=samples,t=t,)
            if mask is not None and i > 0:
                t_prev = torch.full((num_samples,), i - 1, dtype=torch.long, device=device)
                known_prev = self.q_sample(known_full, t_prev)
                samples = (mask * known_prev) + ((1 - mask) * x_prev_uncond)
            else:
                samples = x_prev_uncond

            
        if plot:
            make_diffusion_gif()
            
        return samples

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for rate, state_dict in zip(self.ema_rate, self.ema_state_dicts):
            update_ema(state_dict, self.backbone.state_dict(), rate=rate)


def update_ema(target_state_dict, source_state_dict, rate=0.99):
    with torch.no_grad():
        for key, value in source_state_dict.items():
            ema_value = target_state_dict[key]
            ema_value.copy_(
                rate * ema_value + (1.0 - rate) * value.cpu(),
                non_blocking=True,
            )
