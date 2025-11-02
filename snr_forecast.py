import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt




SEQ_LEN = 128       
NUM_SAMPLES = 10000 
BATCH_SIZE = 128


TIMESTEPS = 200     


MODEL_DIM = 64      
TIME_DIM = 64 * 4   


EPOCHS = 100 
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# This is the 'v_t' noise in y_t = x_t + v_t
OBSERVATION_NOISE_STD = 0.2

print(f"Using device: {DEVICE}")



def generate_sine_waves(num_samples, seq_len):
    """
    Generates a dataset of clean "state" (x_t) sine waves.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 1, seq_len)
    """
    x = torch.zeros((num_samples, seq_len))
    y = torch.zeros((num_samples,seq_len))
    t = torch.linspace(0, 10 * np.pi, seq_len)

    for i in range(num_samples):
        
        amplitude = torch.rand(1) * 1.8 + 0.2  
        frequency = torch.rand(1) * 1.5 + 0.5  
        phase = torch.rand(1) * 2 * np.pi      
        x[i, :] = amplitude * torch.sin(frequency * t + phase)
        y[i, :] = x[i, :] + torch.randn(seq_len) * OBSERVATION_NOISE_STD

    
    return x.unsqueeze(1), y.unsqueeze(1)

def dual_sine_waves(num_samples, seq_len):
    x = torch.zeros((num_samples, seq_len))
    y = torch.zeros((num_samples,seq_len))
    t = torch.linspace(0, 10 * np.pi, seq_len)

    for i in range(num_samples):
        
        amplitude1 = torch.rand(1) * 1.8 + 0.2  
        frequency1 = torch.rand(1) * 1.5 + 0.5  
        phase1 = torch.rand(1) * 2 * np.pi 
        amplitude2 = torch.rand(1) * 1.8 + 0.2  
        frequency2 = torch.rand(1) * 1.5 + 0.5  
        phase2 = torch.rand(1) * 2 * np.pi  
             
        x[i, :] = amplitude1 * torch.sin(frequency1 * t + phase1) + amplitude2 * torch.sin(frequency2 * t + phase2)
        y[i, :] = x[i, :] + torch.randn(seq_len) * OBSERVATION_NOISE_STD

    
    return x.unsqueeze(1), y.unsqueeze(1)

class Diffusion:
    def __init__(self, timesteps=TIMESTEPS, beta_start=0.0001, beta_end=0.02, device=DEVICE):
        self.timesteps = timesteps
        self.device = device

        
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        # Pre-calculate diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for reverse step (p_sample)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        """Extracts coefficients for a given batch of timesteps t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward process: q(x_t | x_0)
        Noises x_start to timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        Reverse process: p(x_{t-1} | x_t)
        Denoises x_t to x_{t-1} using the model.
        """
        
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x_t.shape)

        predicted_noise = model(x_t, t)

        
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean # No noise added at the last step

        
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t, device=self.device)

        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Full sampling loop from T to 0 (Unconditional Generation).
        """
        img = torch.randn(shape, device=self.device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=self.device)
            img = self.p_sample(model, img, t)

        return img

    @torch.no_grad()
    def p_sample_loop_from_t(self, model, x_t, start_t):
        """
        Denoising loop from a specific timestep t down to 0 (Denoising/Recovery).
        """
        img = x_t.to(self.device)
        for i in reversed(range(0, start_t)):
            t = torch.full((img.shape[0],), i, dtype=torch.long, device=self.device)
            img = self.p_sample(model, img, t)
        return img

    @torch.no_grad()
    def p_sample_loop_with_snr(self, model, x_0_original, x_t_start, start_t):
        """
        Denoising loop that records SNR at each step.
        x_0_original is the clean signal, for comparison.
        x_t_start is the noisy signal at start_t.
        """
        snr_list = []
        t_list = []
        intermediate_waves = {}

        img = x_t_start.to(self.device)
        x_0_original_gpu = x_0_original.to(self.device)

        
        
        P_signal = torch.mean(x_0_original_gpu ** 2)

        mid_t = start_t // 2

        for i in reversed(range(0, start_t)): 
            
            
            error = img - x_0_original_gpu
            P_noise = torch.mean(error ** 2)
            snr = 10 * torch.log10(P_signal / (P_noise + 1e-9))
            snr_list.append(snr.item())
            t_list.append(i + 1)

            
            if (i + 1) == start_t:
                intermediate_waves[start_t] = img.cpu().numpy()
            if (i + 1) == mid_t:
                intermediate_waves[mid_t] = img.cpu().numpy()

            
            t_tensor = torch.full((img.shape[0],), i, dtype=torch.long, device=self.device)
            img = self.p_sample(model, img, t_tensor) 

        
        intermediate_waves[0] = img.cpu().numpy()
        
        error = img - x_0_original_gpu
        P_noise = torch.mean(error ** 2)
        snr = 10 * torch.log10(P_signal / (P_noise + 1e-9))
        snr_list.append(snr.item())
        t_list.append(0)

        return t_list, snr_list, intermediate_waves

    @torch.no_grad()
    def p_sample_loop_forecasting(self, model, x_known, known_len):
        """
        Performs forecasting (in-painting) by conditioning on x_known.
        This function now supports the SSM case, where x_known is the
        *noisy observation y_known*. The RePaint algorithm handles this.
        """
        batch_size = x_known.shape[0]
        shape = (batch_size, 1, SEQ_LEN)

        
        mask = torch.zeros(shape, device=self.device)
        mask[:, :, :known_len] = 1

        
        x_known_full = torch.zeros(shape, device=self.device)
        x_known_full[:, :, :known_len] = x_known

        
        img = torch.randn(shape, device=self.device) # This is x_T

        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=self.device)
            x_prev_uncond = self.p_sample(model, img, t)

            if i > 0:
                
                
                
                
                t_prev = torch.full((batch_size,), i - 1, dtype=torch.long, device=self.device)
                known_prev = self.q_sample(x_known_full, t_prev)

                
                
                img = (mask * known_prev) + ((1 - mask) * x_prev_uncond)
            else:
                
                img = x_prev_uncond

        return img



class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes timestep t into a vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    """Standard 1D Convolutional Block."""
    def __init__(self, in_channels, out_channels, mid_channels=None, time_emb_dim=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim else None

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, mid_channels)
        self.relu = nn.SiLU()
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, time_emb=None):
        h = self.conv1(x)
        h = self.norm(h)

        if self.mlp and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = h + time_emb.unsqueeze(-1) 

        h = self.relu(h)
        return self.conv2(h)

class DownBlock(nn.Module):
    """Downsampling block."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim=time_emb_dim)
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.SiLU()

    def forward(self, x, time_emb):
        x = self.conv(x, time_emb)
        skip = x
        x = self.relu(self.downsample(x))
        return x, skip

class UpBlock(nn.Module):
    """Upsampling block."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim=time_emb_dim) 
        self.relu = nn.SiLU()

    def forward(self, x, skip, time_emb):
        x = self.relu(self.upsample(x))
        x = torch.cat((skip, x), dim=1) 
        x = self.conv(x, time_emb)
        return x

class Unet1D(nn.Module):
    """
    A 1D U-Net model for denoising time-series data.
    """
    def __init__(self, model_dim=MODEL_DIM, in_channels=1, out_channels=1, time_emb_dim=TIME_DIM):
        super().__init__()

        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_dim),
            nn.Linear(model_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        
        self.init_conv = nn.Conv1d(in_channels, model_dim, kernel_size=3, padding=1)

        self.down1 = DownBlock(model_dim, model_dim * 2, time_emb_dim)
        self.down2 = DownBlock(model_dim * 2, model_dim * 4, time_emb_dim)

        
        self.mid_block = ConvBlock(model_dim * 4, model_dim * 8, time_emb_dim=time_emb_dim)

        
        self.up1 = UpBlock(model_dim * 8, model_dim * 4, time_emb_dim)
        self.up2 = UpBlock(model_dim * 4, model_dim * 2, time_emb_dim)

        
        self.final_conv = nn.Sequential(
            ConvBlock(model_dim * 2, model_dim, time_emb_dim=time_emb_dim),
            nn.Conv1d(model_dim, out_channels, kernel_size=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.init_conv(x) 

        x, skip1 = self.down1(x, t_emb) 
        x, skip2 = self.down2(x, t_emb) 

        x = self.mid_block(x, t_emb)    

        x = self.up1(x, skip2, t_emb)   
        x = self.up2(x, skip1, t_emb)   

        return self.final_conv(x)      



def train(generator):
    print("Generating data...")
    state, obs = generator(NUM_SAMPLES, SEQ_LEN)
    dataloader = torch.utils.data.DataLoader(obs, batch_size=BATCH_SIZE, shuffle=True)

    print("Initializing model and diffusion...")
    model = Unet1D().to(DEVICE)
    diffusion = Diffusion(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            x_start = batch.to(DEVICE)
            batch_size = x_start.shape[0]

            
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()

            
            noise = torch.randn_like(x_start, device=DEVICE)
            x_t = diffusion.q_sample(x_start, t, noise)

            
            predicted_noise = model(x_t, t)

            
            loss = F.mse_loss(predicted_noise, noise)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f}")

    print("Training complete.")
    return model, diffusion, state, obs



def plot_results(model, diffusion, real_data, obs_data, num_to_generate=4):
    print("Generating new sine waves...")

    
    generated_samples = diffusion.p_sample_loop(
        model,
        shape=(num_to_generate, 1, SEQ_LEN)
    )
    generated_samples = generated_samples.cpu().numpy()

    
    real_samples = real_data[:num_to_generate].numpy()

    print("Plotting results...")
    plt.figure(figsize=(12, 8))

    time_axis = np.arange(SEQ_LEN)

    for i in range(num_to_generate):
        
        plt.subplot(num_to_generate, 2, 2*i + 1)
        plt.plot(time_axis, real_samples[i, 0, :],label='True Signal')
        plt.plot(time_axis, obs_data[i,0,:],'.',label='Noisy Observation',alpha=0.5)
        plt.title(f"Training Sample {i+1}")
        plt.ylim(-2.1, 2.1)

        
        plt.subplot(num_to_generate, 2, 2*i + 2)
        plt.plot(time_axis, generated_samples[i, 0, :])
        plt.title(f"Generated Sample {i+1}")
        plt.ylim(-2.1, 2.1)

    plt.tight_layout()
    plt.savefig("sine_wave_generation.png")
    print("Plot saved to sine_wave_generation.png")
    plt.show()

def plot_denoising_results(model, diffusion, real_data, num_to_denoise=3, noise_level_t=100):
    """
    Takes a clean sample, noises it to t, then denoises it and plots all three.
    """
    print(f"Denoising test samples from t={noise_level_t}...")

    
    x_0_samples = real_data[:num_to_denoise].to(DEVICE)
    num_samples = x_0_samples.shape[0]

    
    t = torch.full((num_samples,), noise_level_t, dtype=torch.long, device=DEVICE)
    noise = torch.randn_like(x_0_samples, device=DEVICE)
    x_t_samples = diffusion.q_sample(x_0_samples, t, noise)

    
    recovered_x_0_samples = diffusion.p_sample_loop_from_t(
        model,
        x_t_samples,
        noise_level_t
    )

    # Move to CPU for plotting
    x_0_samples = x_0_samples.cpu().numpy()
    x_t_samples = x_t_samples.cpu().numpy()
    recovered_x_0_samples = recovered_x_0_samples.cpu().numpy()

    print("Plotting denoising results...")
    plt.figure(figsize=(12, num_to_denoise * 3))
    time_axis = np.arange(SEQ_LEN)

    for i in range(num_to_denoise):
        plt.subplot(num_to_denoise, 3, 3*i + 1)
        plt.plot(time_axis, x_0_samples[i, 0, :])
        plt.title(f"Sample {i+1}: Original (x_0)")
        plt.ylim(-2.1, 2.1)

        plt.subplot(num_to_denoise, 3, 3*i + 2)
        plt.plot(time_axis, x_t_samples[i, 0, :])
        plt.title(f"Sample {i+1}: Noisy (x_{noise_level_t})")
        plt.ylim(-2.1, 2.1)

        plt.subplot(num_to_denoise, 3, 3*i + 3)
        plt.plot(time_axis, recovered_x_0_samples[i, 0, :])
        plt.title(f"Sample {i+1}: Recovered")
        plt.ylim(-2.1, 2.1)

    plt.tight_layout()
    plt.savefig("sine_wave_denoising.png")
    print("Denoising plot saved to sine_wave_denoising.png")
    plt.show()

def plot_snr_and_recovery_steps(model, diffusion, real_data, noise_level_t=100):
    """
    Plots the SNR increase during denoising and shows intermediate recovery steps.
    """
    print(f"Plotting SNR and recovery steps from t={noise_level_t}...")

    
    x_0_sample = real_data[0:1].to(DEVICE) 

    
    t = torch.full((1,), noise_level_t, dtype=torch.long, device=DEVICE)
    noise = torch.randn_like(x_0_sample, device=DEVICE)
    x_t_sample = diffusion.q_sample(x_0_sample, t, noise)

    
    t_list, snr_list, intermediate_waves = diffusion.p_sample_loop_with_snr(
        model,
        x_0_sample,
        x_t_sample,
        noise_level_t
    )

    print("Plotting SNR curve and intermediate steps...")
    plt.figure(figsize=(16, 8))

    
    plt.subplot(2, 1, 1)
    plt.plot(t_list, snr_list)
    plt.title(f"Signal-to-Noise Ratio (SNR) During Denoising")
    plt.xlabel("Denoising Timestep (t)")
    plt.ylabel("SNR (dB)")
    plt.xlim(noise_level_t, 0) 
    plt.grid(True)

    
    time_axis = np.arange(SEQ_LEN)

    
    plt.subplot(2, 4, 5)
    plt.plot(time_axis, x_0_sample.cpu().numpy()[0, 0, :])
    plt.title("1. Original (x_0)")
    plt.ylim(-2.1, 2.1)

    
    start_t_wave = intermediate_waves.get(noise_level_t)
    if start_t_wave is not None:
        plt.subplot(2, 4, 6)
        plt.plot(time_axis, start_t_wave[0, 0, :])
        plt.title(f"2. Noisy (x_{noise_level_t})")
        plt.ylim(-2.1, 2.1)

    
    mid_t = noise_level_t // 2
    mid_wave = intermediate_waves.get(mid_t)
    if mid_wave is not None:
        plt.subplot(2, 4, 7)
        plt.plot(time_axis, mid_wave[0, 0, :])
        plt.title(f"3. Mid-Denoising (x_{mid_t})")
        plt.ylim(-2.1, 2.1)

    
    recovered_wave = intermediate_waves.get(0)
    if recovered_wave is not None:
        plt.subplot(2, 4, 8)
        plt.plot(time_axis, recovered_wave[0, 0, :])
        plt.title("4. Recovered")
        plt.ylim(-2.1, 2.1)

    plt.tight_layout()
    plt.savefig("sine_wave_snr_recovery.png")
    print("SNR and recovery plot saved to sine_wave_snr_recovery.png")
    plt.show()

def plot_forecasting_results(model, diffusion, real_data, num_to_forecast=3, observation_noise_std=0.1):
    """
    MODIFIED: Plots forecasting as a state-space model problem.
    It takes the *clean state* (real_data, x_t), creates a *noisy observation* (y_t),
    and then uses the first half of y_t to forecast the second half of x_t.
    """
    print("Running forecasting (SSM inference)...")

    known_len = SEQ_LEN // 2 
    time_axis = np.arange(SEQ_LEN)

    
    x_0_originals = real_data[:num_to_forecast].to(DEVICE)

    
    noise = torch.randn_like(x_0_originals) * observation_noise_std
    y_0_observations = x_0_originals + noise

    
    y_knowns = y_0_observations[:, :, :known_len]

    
    forecasted_states = diffusion.p_sample_loop_forecasting(
        model,
        y_knowns,
        known_len
    )

    
    x_0_originals_cpu = x_0_originals.cpu().numpy()
    y_0_observations_cpu = y_0_observations.cpu().numpy()
    forecasted_states_cpu = forecasted_states.cpu().numpy()

    print("Plotting forecasting results...")
    plt.figure(figsize=(12, num_to_forecast * 4))

    for i in range(num_to_forecast):
        plt.subplot(num_to_forecast, 1, i + 1)

        
        plt.plot(time_axis, x_0_originals_cpu[i, 0, :], label="Ground Truth (State $x_t$)", color='blue', linewidth=2)

        
        plt.plot(time_axis, y_0_observations_cpu[i, 0, :], label="Noisy Observation ($y_t$)", color='green', alpha=0.5)

        
        plt.plot(time_axis, forecasted_states_cpu[i, 0, :], label="Forecast (Recovered State $\\hat{x}_t$)", color='orange', linestyle='--', linewidth=2)

        
        plt.axvline(x=known_len - 0.5, color='red', linestyle=':', label="Forecast Start")

        plt.title(f"Forecast Example {i+1} (SSM)")
        plt.ylim(-2.1, 2.1)
        plt.legend()

    plt.tight_layout()
    plt.savefig("sine_wave_forecasting.png")
    print("Forecasting plot saved to sine_wave_forecasting.png")
    plt.show()




if __name__ == "__main__":
    
    trained_model, diffusion_helper, real_data, obs_data = train(dual_sine_waves)

    
    plot_results(trained_model, diffusion_helper, real_data, obs_data, num_to_generate=4)

    
    denoise_t = 3 * TIMESTEPS // 4 
    plot_denoising_results(
        trained_model,
        diffusion_helper,
        real_data,
        num_to_denoise=3,
        noise_level_t=denoise_t
    )

    
    plot_snr_and_recovery_steps(
        trained_model,
        diffusion_helper,
        real_data,
        noise_level_t=denoise_t
    )

    
    plot_forecasting_results(
        trained_model,
        diffusion_helper,
        real_data,
        num_to_forecast=3,
        observation_noise_std=OBSERVATION_NOISE_STD
    )