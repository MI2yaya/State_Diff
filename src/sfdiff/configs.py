# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from sfdiff.utils import linear_beta_schedule

#backbone

backbone_s5 = {
    "hidden_dim": 64,
    "time_emb_dim": 256,
    "num_residual_blocks": 5,
}
backbone_s5_large = {
    "hidden_dim": 64,
    "time_emb_dim": 256,
    "num_residual_blocks": 10,
}
backbone_s4 = {
    "hidden_dim": 64,
    "time_emb_dim": 256,
    "num_residual_blocks": 5,
}
backbone_unet = {
    "hidden_dim": 64,
    "time_emb_dim": 256,
}

#diff

diffusion_s4 = {
    "backbone_parameters": backbone_s4,
    "timesteps": 200,
    "diffusion_scheduler": linear_beta_schedule,
}
diffusion_s5 = {
    "backbone_parameters": backbone_s5,
    "timesteps": 200,
    "diffusion_scheduler": linear_beta_schedule,
}
diffusion_s5_large = {
    "backbone_parameters": backbone_s5_large,
    "timesteps": 200,
    "diffusion_scheduler": linear_beta_schedule,
}
diffusion_unet = {
    "backbone_parameters": backbone_unet,
    "timesteps": 200,
    "diffusion_scheduler": linear_beta_schedule,
}