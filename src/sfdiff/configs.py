# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from sfdiff.utils import linear_beta_schedule


s4_backbone = {
    "hidden_dim": 64,
    "time_emb_dim": 256,
    "num_residual_blocks": 5,
}
unet_backbone = {
    "hidden_dim": 64,
    "time_emb_dim": 256,
}
diffusion_s4 = {
    "backbone_parameters": s4_backbone,
    "timesteps": 200,
    "diffusion_scheduler": linear_beta_schedule,
}
diffusion_unet = {
    "backbone_parameters": unet_backbone,
    "timesteps": 200,
    "diffusion_scheduler": linear_beta_schedule,
}