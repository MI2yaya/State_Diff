# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import torch
from torch import nn

from .s4 import S4
from .s5 import S5

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SequenceLayer(nn.Module):
    """
    Wraps either S4Layer or S5Layer with identical API.
    """
    def __init__(self, d_model, dropout, block_type="s4"):
        super().__init__()
        self.block_type = block_type.lower()

        if self.block_type == "s4":
            self.layer = S4(
                d_model=d_model,
                d_state=128,
                bidirectional=True,
                dropout=dropout,
                transposed=True,
                postact=None,
            )
        elif self.block_type == "s5":
            self.layer = S5(
                d_model=d_model,
                d_state=64,
                bidirectional=True,
                dropout=dropout,
                transposed=True,
                postact=None,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        self.norm = nn.LayerNorm(d_model)
        self.dropout = (
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        """
        x: (B, d_model, L)
        """
        z = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        z, _ = self.layer(z)
        z = self.dropout(z)
        return x + z, None  # residual

    def step(self, x, state):
        # optional recurrent mode, identical for both S4 & S5
        z = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        z, state = self.layer.step(z, state)
        return x + z, state
    
    
    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)



class SequenceBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0, expand=2,block_type='s4'):
        super().__init__()
        # S4Layer already handles dropout internally
        self.core = SequenceLayer(d_model, dropout, block_type)
        self.time_linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.out_linear1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1
        )
        self.out_linear2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1
        )
        self.additional_dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, t):
        t = self.time_linear(t)[:, None, :].repeat(1, x.shape[2], 1)
        t = t.transpose(-1, -2)
        out, _ = self.core(x + t)  # S4Layer handles dropout internally
        out = self.tanh(out) * self.sigm(out)
        
        # Apply additional dropout after activation but before final layers
        out = self.additional_dropout(out)
        
        out1 = self.out_linear1(out)
        out2 = self.out_linear2(out)
        return out1 + x, out2


def Conv1dKaiming(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class SequenceBackbone(nn.Module):
    def __init__(
        self,
        observation_dim,
        hidden_dim,
        output_dim,
        time_emb_dim,
        num_residual_blocks,
        dropout=0.0,
        init_skip=True,
        block_type="s4",
    ):
        super().__init__()
        self.block_type = block_type.lower()
        self.input_init = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
        )
        self.time_init = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual_blocks = nn.ModuleList([
            SequenceBlock(
                hidden_dim,
                dropout=dropout,
                block_type=block_type,
            )
            for _ in range(num_residual_blocks)
        ])
        self.step_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.init_skip = init_skip

    def forward(self, input, t):
        x = self.input_init(input)  # B, L ,C
        t = self.time_init(self.step_embedding(t))
        x = x.transpose(-1, -2)
        skips = []
        for layer in self.residual_blocks:
            x, skip = layer(x, t)
            skips.append(skip)

        skip = torch.stack(skips).sum(0)
        skip = skip.transpose(-1, -2)
        out = self.out_linear(skip)
        if self.init_skip:
            out = out + input
        return out

class Conv1d(nn.Module):
    """Conv1d wrapper that keeps [B, L, C] layout."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, transpose=False):
        super().__init__()
        if not transpose:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.transpose = transpose

    def forward(self, x):
        # x: [B, L, C] → [B, C, L]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim else None

        self.conv1 = Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.SiLU()
        self.conv2 = Conv1d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, time_emb=None):
        h = self.conv1(x)
        h = self.norm(h)
        if self.mlp and time_emb is not None:
            h = h + self.mlp(time_emb).unsqueeze(1)
        h = self.relu(h)
        return self.conv2(h)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.SiLU()

    def forward(self, x, time_emb):
        x = self.conv(x, time_emb)
        skip = x
        x = self.relu(self.downsample(x))
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.upsample = Conv1d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1, transpose=True)
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.relu = nn.SiLU()

    def forward(self, x, skip, time_emb):
        x = self.relu(self.upsample(x))
        x = torch.cat((skip, x), dim=-1)  # concat along channel dim (last)
        x = self.conv(x, time_emb)
        return x


class UNetBackbone(nn.Module):
    def __init__(self, observation_dim, hidden_dim, output_dim, time_emb_dim):

        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = Conv1d(observation_dim, hidden_dim, 3, padding=1)
        self.down1 = DownBlock(hidden_dim, hidden_dim * 2, time_emb_dim)
        self.down2 = DownBlock(hidden_dim * 2, hidden_dim * 4, time_emb_dim)
        self.mid_block = ConvBlock(hidden_dim * 4, hidden_dim * 8, time_emb_dim)
        self.up1 = UpBlock(hidden_dim * 8, hidden_dim * 4, time_emb_dim)
        self.up2 = UpBlock(hidden_dim * 4, hidden_dim * 2, time_emb_dim)
        self.final_conv = nn.Sequential(
            ConvBlock(hidden_dim * 2, hidden_dim, time_emb_dim),
            Conv1d(hidden_dim, output_dim, 1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x = self.mid_block(x, t_emb)
        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        out = self.final_conv(x)
        return out  # stays [B, L, C]