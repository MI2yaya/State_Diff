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

from sfdiff.utils import (
    train_test_val_splitter,
    time_splitter,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateForecastPlotter:
    """Generates state forecasts from SFDiff models and plots them."""
    
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = config['device']
        self.checkpoint_path = checkpoint_path
    
    def _load_model(self, checkpoint_path: str,h_fn,R_inv):
        logger.info(f"Loading model checkpoint: {Path(checkpoint_path).name}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        scaling = int(self.config['dt']**-1)
        context_length = self.config["context_length"] * scaling
        prediction_length = self.config["prediction_length"] * scaling

        model = SFDiff(
            **getattr(diffusion_configs, self.config["diffusion_config"]),
            observation_dim=self.config["observation_dim"],
            context_length=context_length,
            prediction_length=prediction_length,
            lr=self.config["lr"],
            init_skip=self.config["init_skip"],
            h_fn=h_fn,
            R_inv=R_inv,
            modelType=self.config['diffusion_config'].split('_')[1],
        )

        # Load state_dict
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys in state_dict: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in state_dict: {unexpected}")
        
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def generate_state_forecasts(self, dataset_name: str, start_index=0, num_series=1, num_samples=100):
        """Generate forecasts for multiple time series."""
        scaling = int(self.config['dt']**-1)
        testing_samples = num_series-start_index
        
        dataset, generator = get_custom_dataset(dataset_name,
            samples=testing_samples,
            context_length=self.config["context_length"],
            prediction_length=self.config["prediction_length"],
            dt=self.config['dt'],
            q=self.config['q'],
            r=self.config['r'],
            observation_dim=self.config['observation_dim'],
            plot=False
            )
        
        self.model = self._load_model(self.checkpoint_path,generator.h_fn,generator.R_inv)

        selected_series = dataset[start_index:(start_index + num_series)]
        state_series = time_splitter(selected_series, self.config["context_length"] * scaling, self.config["prediction_length"] * scaling)
        forecasts = []
        for series in state_series:

            past_observation = torch.as_tensor(series["past_observation"], dtype=torch.float32)

            if past_observation.ndim == 2:  # shape (batch, seq_len, dims)
                past_observation = past_observation.unsqueeze(0) 


            y = past_observation.to(device=self.model.device, dtype=torch.float32)
            # Generate samples from model
            generated= self.model.sample_n(
                y=y,
                x_known=past_observation, 
                known_len=past_observation.shape[1],
                num_samples=num_samples,
                cheap=False,
                base_strength=.5,
                plot=False,
                guidance=True,
            )
    
            forecasts.append(generated.cpu().numpy())
            #break # Remove this break to process all series

        return forecasts, state_series

    def plot_forecast(self, forecast, series_data, ax, title="Forecast", dim_idx = 0):
        """Plot a single forecast against ground truth states."""
        past_state = series_data["past_state"]
        future_state = series_data["future_state"]
        
        total_state = np.concatenate([past_state, future_state], axis=0)
        
        past_observation = series_data['past_observation']
        future_observation = series_data['future_observation']
        total_obs = np.concatenate([past_observation, future_observation], axis=0)


        # Check if data is 3D
        is_3d = forecast.shape[1] == 3


        if is_3d:

            forecast_vals=forecast
            
            # 3D trajectory plot
            ax_3d = ax if hasattr(ax, "plot") else ax.figure.add_subplot(111, projection='3d')
            ax_3d.plot(total_state[:, 0], total_state[:, 1], total_state[:, 2], color='#1f77b4', linewidth=2.5, label='Ground Truth')
            ax_3d.plot(forecast_vals.mean(axis=0)[:, 0], forecast_vals.mean(axis=0)[:, 1], forecast_vals.mean(axis=0)[:, 2],
                    color='black', linewidth=2, label='Median Forecast', linestyle='--')
            ax_3d.set_xlabel("X")
            ax_3d.set_ylabel("Y")
            ax_3d.set_zlabel("Z")
            ax_3d.set_title(title)
            ax_3d.legend()
        else:
            forecast_vals = forecast[:, :, 0] 
            quartiles = forecast_vals.shape[0] > 1
            
            dataRange = np.arange(0, self.config['prediction_length'] + self.config['context_length'], self.config['dt'])

            # Plot ground truth
            ax.plot(dataRange, total_state[:, dim_idx], color='#1f77b4', linewidth=2.5, label="Ground Truth")  # bright blue

            # Median forecast
            median_forecast = np.median(forecast_vals, axis=0)
            ax.plot(dataRange, median_forecast, color='black', linewidth=2, label="Median Forecast")

            # Plot observations only up to context length
            obs_end_idx = int(self.config['context_length'] / self.config['dt'])
            ax.plot(dataRange[:obs_end_idx], total_obs[:obs_end_idx, dim_idx],
                    color='#ff7f0e', linewidth=1.5, linestyle='--', label="Observation", dashes=(8, 8))

            # Shaded forecast intervals
            if quartiles:
                lower_90 = np.quantile(forecast_vals, 0.05, axis=0)
                upper_90 = np.quantile(forecast_vals, 0.95, axis=0)
                lower_50 = np.quantile(forecast_vals, 0.25, axis=0)
                upper_50 = np.quantile(forecast_vals, 0.75, axis=0)
                ax.fill_between(dataRange, lower_90, upper_90, color='indianred', alpha=0.25, label='90% Interval')
                ax.fill_between(dataRange, lower_50, upper_50, color='red', alpha=0.45, label='50% Interval')

            ax.axvline(x=self.config['context_length'], color='gray', linestyle=':', alpha=0.8)
            ax.set_title(title)
            ax.grid(True, alpha=0.4)
            ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
        


def create_continual_learning_plots(config, start_series=0, num_series=3):
    """Create plots for multiple time series."""
    checkpoints = {
        "Single Task": config["checkpoint_path"]
    }

    obs_dim = config['observation_dim']
    fig, axes = plt.subplots(
        num_series, obs_dim, figsize=(6 * obs_dim, 4 * num_series), squeeze=False
    )
    # Ensure axes is always 2D
    for method_idx, (method_name, ckpt_path) in enumerate(checkpoints.items()):
        plotter = StateForecastPlotter(config, ckpt_path)
        forecasts, series_list = plotter.generate_state_forecasts(
            config["dataset"], start_index=start_series, num_series=num_series, num_samples=10
        )

        for series_idx, forecast in enumerate(forecasts):
            series_data = series_list[series_idx]
            for dim_idx in range(obs_dim):
                ax = axes[series_idx, dim_idx]

                # Extract only this dimension from forecast for plotting
                forecast_dim = forecast[:, :, dim_idx]
                plotter.plot_forecast(
                    forecast_dim[:, :, None],  # restore shape for consistency
                    series_data,
                    ax,
                    title=f"TS{start_series + series_idx} (Dim {dim_idx + 1})",
                    dim_idx=dim_idx
                )
            
        logger.info(f"✓ {method_name} plots completed")

    plt.tight_layout()
    plt.savefig(
        f"continual_learning_states_{start_series}_to_{start_series+num_series-1}_{config['dataset'].replace(':','_')}.png",
        dpi=300
    )
    plt.close(fig)
    logger.info("Comparison plots saved.")

def main(config_path):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    create_continual_learning_plots(config, start_series=0, num_series=3)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    args = parser.parse_args()

    main(args.config)
