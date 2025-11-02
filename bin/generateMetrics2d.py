
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import math

import sfdiff.configs as diffusion_configs
from sfdiff.model.diffusion.diff import SFDiff
from sfdiff.dataset import get_custom_dataset
from sfdiff.utils import time_splitter, train_test_val_splitter
from dataGeneration import make_kf_matrices_for_sinusoid

from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def mse(a, b):
    mse = 0
    print(a.shape, b.shape)
    raise ValueError
    assert a.shape[2]==b.shape[2]
    for a_feat,b_feat in zip(a[1],b[1]):
        mse+=np.mean((a_feat-b_feat) ** 2)
    return mse

def crps(pred, true):
    pred = torch.as_tensor(pred, dtype=torch.float32)
    true = torch.as_tensor(true, dtype=torch.float32)

    # CASE: both 1-D and same length -> treat as sequence batch
    if pred.ndim == 1 and true.ndim == 1 and pred.shape[0] == true.shape[0]:
        # turn into [B=T, N=1, F=1] and [B=T, 1, 1]
        pred = pred.view(-1, 1, 1)
        true = true.view(-1, 1, 1)
    else:
        # Generic promotion to [B, N, F] and [B, 1, F]
        if pred.ndim == 1:
            pred = pred.unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
        elif pred.ndim == 2:
            pred = pred.unsqueeze(-1)               # [B, N, 1]
        # true handling
        if true.ndim == 0:
            true = true.unsqueeze(0).unsqueeze(-1)  # scalar -> [1,1,1]
        elif true.ndim == 1:
            true = true.unsqueeze(1).unsqueeze(-1)  # [B,1,1]  (if B matches pred batch)
        elif true.ndim == 2:
            true = true.unsqueeze(1)               # [B,1,F]

    # Now pred: [B, N, F], true: [B, 1, F]
    # Compute CRPS = E|X - y| - 0.5 E|X - X'|
    term1 = torch.mean(torch.abs(pred - true), dim=1)          # [B, F]
    diffs = torch.abs(pred.unsqueeze(2) - pred.unsqueeze(1))   # [B, N, N, F]
    term2 = 0.5 * torch.mean(diffs, dim=(1, 2))               # [B, F]
    crps_per_feature = term1 - term2
    # clamp tiny negative numeric errors to 0
    crps_per_feature = torch.clamp(crps_per_feature, min=0.0)
    return torch.mean(crps_per_feature).item()


class SimpleKalmanFilter:
    def __init__(self, A, H, Q, R):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

    def predict(self, x, P):
        x_pred = self.A @ x
        P_pred = self.A @ P @ self.A.T + self.Q
        return x_pred, P_pred
    
    def update(self, x_pred, P_pred, y):
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)
        x_upd = x_pred + K @ (y - self.H @ x_pred)
        P_upd = (np.eye(self.A.shape[0]) - K @ self.H) @ P_pred
        return x_upd, P_upd

    def loop(self, y, x0=None, P0=None):
        T = y.shape[0]
        n = self.A.shape[0]
        xs = np.zeros((T, n))

        if x0 is None:
            x = np.zeros((n,))
        else:
            x = x0

        if P0 is None:
            P = np.eye(n)
        else:
            P = P0

        for t in range(T):
            # Predict
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q

            # Update
            x, P = self.update(x_pred, P_pred, y[t])

            xs[t] = x

        self.x = x
        self.P = P
        
        return xs



class SFDiffForecaster:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = config["device"]
        self.checkpoint_path = checkpoint_path
        self.model = None

    def _load_model(self, h_fn, R_inv):
        logger.info(f"Loading model checkpoint: {Path(self.checkpoint_path).name}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        scaling = int(self.config["dt"] ** -1)
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

        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()
        self.model = model
        return(model)
        
    def p_sample_loop_with_snr(self,x_0,num_samples=10):
        start_t = self.model.timesteps
        t = torch.full((num_samples,),start_t-1, dtype=torch.long, device=self.device)
        noise = torch.randn_like(x_0).to(self.device)
        x_0 = x_0.to(self.device)   
        
        x_t = self.model.q_sample(x_0, t, noise=noise)
        
        snr_list = []
        t_list = []
        intermediate_waves = {}

        x_t = x_t.to(self.device)
        x_0 = x_0.to(self.device)
        
        P_signal = torch.mean(x_0 ** 2)
        mid_t = start_t // 2

        for i in reversed(range(start_t)):
            # Compute SNR
            P_noise = torch.mean((x_t - x_0) ** 2)
            snr = 10 * torch.log10(P_signal / (P_noise + 1e-9))
            snr_list.append(snr.item())
            t_list.append(i + 1)

            # Save intermediate waves
            if (i + 1) == start_t or (i + 1) == mid_t:
                intermediate_waves[i + 1] = x_t.detach().cpu().numpy()

            # Sample next step
            t_step = torch.full((1,), i, dtype=torch.long, device=self.device)
            x_t, _ = self.model.p_sample(x=x_t, t=t_step, t_index=i, guidance=False)


        
        x_med = torch.median(x_t, dim=0).values 
        intermediate_waves[0] = x_t 
        P_noise = torch.mean((x_med-x_0) ** 2) 
        snr = 10 * torch.log10(P_signal / (P_noise + 1e-9)) 
        snr_list.append(snr.item()) 
        t_list.append(0)

        return t_list, snr_list, intermediate_waves
    


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    skipKF=True
    
    skipSFDiff=False
    skipSFDiffSNR=True
    skipSFDiffCond=True
    skipSFDiffCheapCond=True
    skipSFDiffExpensiveCond=False
    
    skipAR = False
    skipARX = False
    skipNARX = False
    
    plot = False


    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f" {config['dataset']}, {config['data_samples']}, with {config['diffusion_config']}")

    dataset_name = config["dataset"]
    scaling = int(config['dt']**-1)

    context_length = config["context_length"] * scaling
    prediction_length = config["prediction_length"] * scaling

    num_data_samples=11500

    base_dataset, generator = get_custom_dataset(dataset_name,
        samples=num_data_samples,
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        dt=config['dt'],
        q=config['q'],
        r=config['r'],
        observation_dim=config['observation_dim'],
        plot=False
        )
    time_data = time_splitter(base_dataset, context_length, prediction_length)
    split_data = train_test_val_splitter(time_data, num_data_samples, 10/11, 1/11, 0.0)
    train_data = split_data['train']
    test_data = split_data['test']
    

    SFDiff = SFDiffForecaster(config, config['checkpoint_path'])
    SFDiff._load_model(generator.h_fn, generator.R_inv)
    
    if skipKF==False:
        trials = 1000
        kf_MSEs=[]
        kf_CRPSs=[]
        for i in range(trials):
            sample = test_data[i]
            y_obs = sample["past_observation"]
            x_true_future = sample["future_state"]
            x_preds = []
            
            A, H, Q, R = make_kf_matrices_for_sinusoid(generator,past_obs=y_obs,mode="osc")
            Q = np.eye(A.shape[0]) * 1e-4 if not np.any(Q) else Q + np.eye(A.shape[0]) * 1e-8
            R = np.eye(H.shape[0]) * 1e-3 if not np.any(R) else R + np.eye(H.shape[0]) * 1e-8
            
            kf = SimpleKalmanFilter(A, H, Q, R)
            _ = kf.loop(y_obs)
            
            for t in range(context_length, context_length + prediction_length):
                if t == context_length:
                    x_init = kf.A @ kf.x
                    P_init = kf.A @ kf.P @ kf.A.T + kf.Q
                x_pred, P_pred = kf.predict(x_init, P_init)
                x_init, P_init = x_pred, P_pred
                x_preds.append(x_pred[0])  # Assuming we're interested in the first state variable

            # Kalman Filter Prediction
            kf_mse= mse(x_preds, x_true_future.squeeze())
            
            kf_crps = crps(x_preds, x_true_future)
            kf_MSEs.append(kf_mse)
            kf_CRPSs.append(kf_crps)
            
            if plot and i==0:
                plt.figure(figsize=(10,5))
                total_length = context_length + prediction_length
                time_axis = np.arange(total_length) * config['dt']
                plt.plot(time_axis[:context_length], y_obs[:context_length], label='Observations', color='blue')
                plt.plot(time_axis[context_length:], x_true_future, label='True Future', color='green')
                plt.plot(time_axis[context_length:], x_preds, label='KF Prediction', color='red')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title('Kalman Filter Prediction vs True Future')
                plt.legend()
                plt.show()
        
        kf_mse_avg = np.mean(kf_MSEs)
        kf_mse_std = np.std(kf_MSEs)
        kf_crps_avg = np.mean(kf_CRPSs)
        kf_crps_std = np.std(kf_CRPSs)
        logger.info(f"KF on all obs: Future MSE={kf_mse_avg:.4f} ± {kf_mse_std:.4f}, CRPS={kf_crps_avg:.4f} ± {kf_crps_std:.4f}")
    

    if skipSFDiff==False:
        if skipSFDiffSNR==False:
            trials = 20
            sfdiff_SNRs=[]
            for i in range(trials):
                sample = test_data[i]
                past_state = torch.as_tensor(sample["past_state"], dtype=torch.float32)
                future_state = torch.as_tensor(sample["future_state"], dtype=torch.float32)
                true_state = torch.cat([past_state, future_state], dim=0).unsqueeze(0)
                
                # SFDiff SNRs
                _, snrs, intermediate_waves = SFDiff.p_sample_loop_with_snr(true_state,num_samples=50)
                sfdiff_SNRs.append(snrs[-1]-snrs[0])
                
                if plot and i==0:
                    fig = plt.figure(figsize=(6,8))
                    ax = fig.add_subplot(411)
    
                    ax.plot(snrs, label='SFDiff SNR', color='orange')
                    ax.legend()
                    
                    ax2 = fig.add_subplot(412)
                    ax2.plot(intermediate_waves[list(intermediate_waves.keys())[0]][0,:,0], label='Initial Noisy Sample', color='red')
                    ax3 = fig.add_subplot(413)
                    ax3.plot(intermediate_waves[list(intermediate_waves.keys())[1]][0,:,0], label='Mid Diffusion Sample', color='green')
                    ax4 = fig.add_subplot(414)
                    ax4.plot(true_state[0,:,0].cpu().numpy(), label='True Signal', color='blue')
                    plt.show()
            
            sfdiff_SNRs_avg = np.mean(sfdiff_SNRs, axis=0)
            sfdiff_SNRs_std = np.std(sfdiff_SNRs, axis=0)
            logger.info(f"SFDiff SNR Change over diffusion steps: mean={sfdiff_SNRs_avg:.4f}, std={sfdiff_SNRs_std:.4f}")
            
        
        if skipSFDiffCond == False:
            trials=10
            batch_size=50
            sfdiff_MSEs=[]
            sfdiff_CRPSs=[]
            for i in range(trials):
    
                sample = test_data[i]
                past_observation = torch.as_tensor(sample["past_observation"], dtype=torch.float32)
                if past_observation.ndim == 2:  # shape (batch, seq_len)
                    past_observation = past_observation.unsqueeze(0) 
                future_state = torch.as_tensor(sample["future_state"], dtype=torch.float32)
                if future_state.ndim == 2:  # shape (batch, seq_len)
                    future_state = future_state.unsqueeze(0)
                y = past_observation.to(device = SFDiff.device,dtype=torch.float32)
                # SFDiff Prediction
                # Generate samples from model
                generated, snr = SFDiff.model.sample_n(
                    y=y,
                    x_known=past_observation, 
                    known_len=past_observation.shape[1],
                    num_samples=batch_size,
                    base_strength=0,
                    plot=False,
                    guidance=False,

                )
                generated_future = np.median(generated[:, -prediction_length:, :].cpu().numpy(),axis=0)
                # Compute MSE and CRPS
                sfdiff_mse = mse(generated_future, future_state.numpy())
                sfdiff_crps = crps(generated[:, -prediction_length:, :].cpu(), future_state)
                sfdiff_MSEs.append(sfdiff_mse)
                sfdiff_CRPSs.append(sfdiff_crps)
                
                if plot and i==0:
                    plt.figure(figsize=(10,5))
                    total_length = past_observation.shape[1] + prediction_length
                    time_axis = np.arange(total_length) * config['dt']
                    plt.plot(time_axis[:past_observation.shape[1]], past_observation.squeeze().cpu().numpy(), label='Past Observations', color='blue')
                    plt.plot(time_axis[past_observation.shape[1]:], future_state.squeeze().cpu().numpy(), label='True Future State', color='green')
                    plt.plot(time_axis[past_observation.shape[1]:], generated_future, label='SFDiff Prediction', color='red')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.title('SFDiff Conditional Prediction vs True Future State')
                    plt.legend()
                    plt.show()
                
            sfdiff_mse_avg = np.mean(sfdiff_MSEs)
            sfdiff_mse_std = np.std(sfdiff_MSEs)
            sfdiff_crps_avg = np.mean(sfdiff_CRPSs)
            sfdiff_crps_std = np.std(sfdiff_CRPSs)
            logger.info(f"SFDiff Conditional: Future MSE={sfdiff_mse_avg:.4f} ± {sfdiff_mse_std:.4f}, CRPS={sfdiff_crps_avg:.4f} ± {sfdiff_crps_std:.4f}")
        
        if skipSFDiffCheapCond == False:
            trials=10
            batch_size=50
            sfdiff_MSEs=[]
            sfdiff_CRPSs=[]
            for i in range(trials):
                sample = test_data[i]
                past_observation = torch.as_tensor(sample["past_observation"], dtype=torch.float32)
                if past_observation.ndim == 2:  # shape (batch, seq_len)
                    past_observation = past_observation.unsqueeze(0) 
                future_state = torch.as_tensor(sample["future_state"], dtype=torch.float32)
                if future_state.ndim == 2:  # shape (batch, seq_len)
                    future_state = future_state.unsqueeze(0)
                y = past_observation.to(device = SFDiff.device,dtype=torch.float32)
                # SFDiff Prediction
                # Generate samples from model
                generated, snr = SFDiff.model.sample_n(
                    y=y,
                    x_known=past_observation, 
                    known_len=past_observation.shape[1],
                    num_samples=batch_size,
                    plot=False,
                    guidance=True,
                    cheap=True,
                    base_strength=0.5

                )
                generated_future = np.median(generated[:, -prediction_length:, :].cpu().numpy(),axis=0)
                # Compute MSE and CRPS
                sfdiff_mse = mse(generated_future, future_state.numpy())
                sfdiff_crps = crps(generated[:, -prediction_length:, :].cpu(), future_state)
                sfdiff_MSEs.append(sfdiff_mse)
                sfdiff_CRPSs.append(sfdiff_crps)
                if plot and i==0:
                    plt.figure(figsize=(10,5))
                    total_length = past_observation.shape[1] + prediction_length
                    time_axis = np.arange(total_length) * config['dt']
                    plt.plot(time_axis[:past_observation.shape[1]], past_observation.squeeze().cpu().numpy(), label='Past Observations', color='blue')
                    plt.plot(time_axis[past_observation.shape[1]:], future_state.squeeze().cpu().numpy(), label='True Future State', color='green')
                    plt.plot(time_axis[past_observation.shape[1]:], generated_future, label='SFDiff Prediction', color='red')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.title('SFDiff Cheap Prediction vs True Future State')
                    plt.legend()
                    plt.show()
                
            sfdiff_mse_avg = np.mean(sfdiff_MSEs)
            sfdiff_mse_std = np.std(sfdiff_MSEs)
            sfdiff_crps_avg = np.mean(sfdiff_CRPSs)
            sfdiff_crps_std = np.std(sfdiff_CRPSs)
            logger.info(f"SFDiff Cheap Guidance + Condition: Future MSE={sfdiff_mse_avg:.4f} ± {sfdiff_mse_std:.4f}, CRPS={sfdiff_crps_avg:.4f} ± {sfdiff_crps_std:.4f}")
            
        if skipSFDiffExpensiveCond == False:
            trials=3
            batch_size=50
            sfdiff_MSEs=[]
            sfdiff_CRPSs=[]
            for i in range(trials):
                sample = test_data[i]
                past_observation = torch.as_tensor(sample["past_observation"], dtype=torch.float32)
                if past_observation.ndim == 2:  # shape (batch, seq_len)
                    past_observation = past_observation.unsqueeze(0) 
                future_state = torch.as_tensor(sample["future_state"], dtype=torch.float32)
                if future_state.ndim == 2:  # shape (batch, seq_len)
                    future_state = future_state.unsqueeze(0)
                y = past_observation.to(device = SFDiff.device,dtype=torch.float32)
                # SFDiff Prediction
                # Generate samples from model
                generated, snr = SFDiff.model.sample_n(
                    y=y,
                    x_known=past_observation, 
                    known_len=past_observation.shape[1],
                    num_samples=batch_size,
                    plot=False,
                    guidance=True,
                    cheap=False,
                    base_strength=0.5,

                )
                generated_future = np.median(generated[:, -prediction_length:, :].cpu().numpy(),axis=0)
                # Compute MSE and CRPS
                sfdiff_mse = mse(generated_future, future_state.numpy())
                sfdiff_crps = crps(generated[:, -prediction_length:, :].cpu(), future_state)
                sfdiff_MSEs.append(sfdiff_mse)
                sfdiff_CRPSs.append(sfdiff_crps)
                if plot and i==0:
                    plt.figure(figsize=(10,5))
                    total_length = past_observation.shape[1] + prediction_length
                    time_axis = np.arange(total_length) * config['dt']
                    plt.plot(time_axis[:past_observation.shape[1]], past_observation.squeeze().cpu().numpy(), label='Past Observations', color='blue')
                    plt.plot(time_axis[past_observation.shape[1]:], future_state.squeeze().cpu().numpy(), label='True Future State', color='green')
                    plt.plot(time_axis[past_observation.shape[1]:], generated_future, label='SFDiff Prediction', color='red')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.title('SFDiff Expensive Prediction vs True Future State')
                    plt.legend()
                    plt.show()
                
            sfdiff_mse_avg = np.mean(sfdiff_MSEs)
            sfdiff_mse_std = np.std(sfdiff_MSEs)
            sfdiff_crps_avg = np.mean(sfdiff_CRPSs)
            sfdiff_crps_std = np.std(sfdiff_CRPSs)
            logger.info(f"SFDiff Expensive Guidance + Condition: Future MSE={sfdiff_mse_avg:.4f} ± {sfdiff_mse_std:.4f}, CRPS={sfdiff_crps_avg:.4f} ± {sfdiff_crps_std:.4f}")
        
        logger.info(f'Done SFDiff')
    
    if not skipAR:
        lag = 10
        ar_MSEs, ar_CRPSs = [], []


        y_train = np.concatenate([s["past_observation"] for s in train_data], axis=0).squeeze()
        model = AutoReg(y_train, lags=lag).fit()


        for i, sample in enumerate(test_data):
            y_obs = sample["past_observation"].squeeze()
            x_true_future = sample["future_state"].squeeze()


            forecast = []
            history = list(y_obs[-lag:]) 

            for _ in range(len(x_true_future)):

                y_pred = model.params[0] 
                for lag_i in range(1, lag+1):
                    y_pred += model.params[lag_i] * history[-lag_i]
                forecast.append(y_pred)
                history.append(y_pred) 

            ar_mse = mse(forecast, x_true_future)
            ar_crps = crps(forecast, x_true_future)
            ar_MSEs.append(ar_mse)
            ar_CRPSs.append(ar_crps)


            if plot and i == 0:
                plt.figure(figsize=(10, 5))
                total_length = len(y_obs) + len(x_true_future)
                time_axis = np.arange(total_length) * config['dt']
                plt.plot(time_axis[:len(y_obs)], y_obs, label='Past Observations', color='blue')
                plt.plot(time_axis[len(y_obs):], x_true_future, label='True Future State', color='green')
                plt.plot(time_axis[len(y_obs):], forecast, label='AR Forecast', color='red')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title('AR Model Prediction vs True Future State')
                plt.legend()
                plt.show()

        logger.info(f"AR({lag}) Model: Future MSE={np.mean(ar_MSEs):.4f} ± {np.std(ar_MSEs):.4f}, "
                    f"CRPS={np.mean(ar_CRPSs):.4f} ± {np.std(ar_CRPSs):.4f}")

    if skipARX == False:
        lag=10
        arx_MSEs, arx_CRPSs = [], []
        model = LinearRegression()


        X_train, Y_train = [], []
        for s in train_data:
            y_obs = np.concatenate([s["past_observation"], s["future_observation"]], axis=0)
            for t in range(lag, len(y_obs)):
                X_train.append(y_obs[t-lag:t].flatten())  # lagged window as input
                Y_train.append(y_obs[t, 0])              # target = next step
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        model.fit(X_train,Y_train)
        # --- Evaluate on test trajectories


        for i,sample in enumerate(test_data):
            forecast = []
            y_obs = sample["past_observation"]
            x_true_future = sample["future_state"].squeeze()

            # start with last 'lag' steps of past observations
            y_hist = y_obs[-lag:].flatten().reshape(1, -1)

            for _ in range(len(x_true_future)):
                y_pred = model.predict(y_hist)[0]
                forecast.append(y_pred)

                # update lag window
                y_hist = np.roll(y_hist, -1)
                y_hist[0, -1] = y_pred


            arx_mse = mse(forecast, x_true_future)
            arx_crps = crps(forecast, x_true_future)
            arx_MSEs.append(arx_mse)
            arx_CRPSs.append(arx_crps)
            
            if plot and i == 0:
                plt.figure(figsize=(10, 5))
                total_length = len(y_obs) + len(x_true_future)
                time_axis = np.arange(total_length) * config['dt']
                plt.plot(time_axis[:len(y_obs)], y_obs.squeeze(), label='Past Observations', color='blue')
                plt.plot(time_axis[len(y_obs):], x_true_future, label='True Future State', color='green')
                plt.plot(time_axis[len(y_obs):], forecast, label='ARX Forecast', color='red')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title('ARX Model Prediction vs True Future State')
                plt.legend()
                plt.show()

        logger.info(f"ARX({lag}) Model: Future MSE={np.mean(arx_MSEs):.4f} ± {np.std(arx_MSEs):.4f}, "
                    f"CRPS={np.mean(arx_CRPSs):.4f} ± {np.std(arx_CRPSs):.4f}")
    
    if skipNARX == False:
        narx_MSEs, narx_CRPSs = [], []
        lag = 10

        # --- Build global train dataset
        X_train, Y_train = [], []
        for s in train_data:
            y_obs = s["past_observation"]
            for t in range(lag, len(y_obs)):
                X_train.append(y_obs[t - lag:t].flatten())
                Y_train.append(y_obs[t, 0])
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
        Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32).unsqueeze(1)

        # --- Train NARX on all trajectories
        model = NARX(input_dim=y_obs.shape[1], lag=lag)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(200):
            model.train()
            opt.zero_grad()
            out = model(X_train)
            loss = loss_fn(out, Y_train)
            loss.backward()
            opt.step()

        # --- Evaluate on each test trajectory
        model.eval()
        for i, sample in enumerate(test_data):
            past_obs = sample["past_observation"]
            x_true_future = sample["future_state"].squeeze()

            # Initialize forecast history with last 'lag' past observations
            y_hist = past_obs[-lag:].copy()
            forecast = []

            # Rollout predictions step-by-step
            for _ in range(len(x_true_future)):
                x_input = torch.tensor(y_hist.flatten(), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    y_pred = model(x_input).item()
                forecast.append(y_pred)

                # Slide window: drop oldest, append newest prediction
                y_hist = np.vstack([y_hist[1:], [y_pred]])

            forecast = np.array(forecast)

            narx_mse = mse(forecast, x_true_future)
            narx_crps = crps(forecast, x_true_future)
            narx_MSEs.append(narx_mse)
            narx_CRPSs.append(narx_crps)

            if plot and i == 0:
                plt.figure(figsize=(10, 5))
                total_length = len(past_obs) + len(x_true_future)
                time_axis = np.arange(total_length) * config['dt']
                plt.plot(time_axis[:len(past_obs)], past_obs.squeeze(), label='Past Observations', color='blue')
                plt.plot(time_axis[len(past_obs):], x_true_future, label='True Future State', color='green')
                plt.plot(time_axis[len(past_obs):], forecast, label='NARX Forecast', color='red')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title('NARX Model Prediction vs True Future State')
                plt.legend()
                plt.show()

        logger.info(f"NARX({lag}) Model: Future MSE={np.mean(narx_MSEs):.4f} ± {np.std(narx_MSEs):.4f}, "
                    f"CRPS={np.mean(narx_CRPSs):.4f} ± {np.std(narx_CRPSs):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    args = parser.parse_args()

    main(args.config)