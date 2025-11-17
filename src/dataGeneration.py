
import numpy as np
import torch
def make_kf_matrices_for_sinusoid(generator, past_obs=None, mode="const_vel"):
    """
    generator: SinusoidalWaves instance (has .dt, .q, .r)
    past_obs: optional 1D numpy array of past observations for frequency estimation
    mode: "const_vel" or "osc"
    Returns: A, H, Q, R (numpy arrays)
    """
    dt = generator.dt
    q = float(generator.q)
    r = float(generator.r)

    if mode == "const_vel":
        A = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        # acceleration spectral density proxy
        q_c = q**2
        Q = q_c * np.array([[dt**3/3.0, dt**2/2.0],
                            [dt**2/2.0, dt]], dtype=float)
        R = np.array([[r**2]], dtype=float)
        return A, H, Q, R

    elif mode == "osc":
        # estimate dominant frequency via FFT
        x = np.asarray(past_obs).astype(float).flatten()
        x = x - x.mean()
        n = len(x)
        freqs = np.fft.rfftfreq(n, dt)  # cycles per second
        X = np.fft.rfft(x)
        idx = np.argmax(np.abs(X))
        f_peak = freqs[idx]
        # protect against zero freq
        f_peak = max(f_peak, 1e-6)
        omega = 2 * np.pi * f_peak

        # oscillator A
        A = np.array([[np.cos(omega*dt), (1.0/omega)*np.sin(omega*dt)],
                      [-omega*np.sin(omega*dt), np.cos(omega*dt)]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        Q = 0.01 * np.eye(2, dtype=float)   # small process noise; tune if needed
        R = np.array([[r**2]], dtype=float)
        return A, H, Q, R

    else:
        raise ValueError("mode must be 'const_vel' or 'osc'")

class SinusoidalWaves():
    def __init__(self,
        length,
        dt,
        q,
        r,
        obs_dim = 1,
        ):
        self.length = length
        self.dt = dt
        self.q = q
        self.r = r
        self.obs_dim = obs_dim
    def h_fn(self, x):
        return x

    def R_inv(self, resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    
    def generate(self):
        xs=[]
        ys=[]
        amplitude=np.random.uniform(0,1) * 1.8 + 0.2 
        frequency=np.random.uniform(0,1) * 1.5 + 0.5  
        phase=np.random.randn(1) * 2 * np.pi
        time_points = np.arange(0, self.length, self.dt)
        for t in time_points:
            x = np.sin(frequency*t + phase)*amplitude + np.random.normal(0,self.q)
            xs.append(x)
            y =  x + np.random.normal(0,self.r)
            ys.append(y)
        return xs, ys
    
class DualSinusoidalWaves():
    def __init__(self,
        length,
        dt,
        q,
        r,
        obs_dim = 1,
        ):
        self.length = length
        self.dt = dt
        self.q = q
        self.r = r
        self.obs_dim = obs_dim
    def h_fn(self, x):
        return x

    def R_inv(self, resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    
    def generate(self):
        xs=[]
        ys=[]
        amplitude1=np.random.uniform(0,1) * 1.8 + 0.2 
        frequency1=np.random.uniform(0,1) * 1.5 + 0.5  
        phase1=np.random.randn(1) * 2 * np.pi
        amplitude2=np.random.uniform(0,1) * 1.8 + 0.2 
        frequency2=np.random.uniform(0,1) * 1.5 + 0.5  
        phase2=np.random.randn(1) * 2 * np.pi
        time_points = np.arange(0, self.length, self.dt)
        for t in time_points:
            x = np.sin(frequency1*t + phase1)*amplitude1 +np.sin(frequency2*t + phase2)*amplitude2 + np.random.normal(0,self.q)
            xs.append(x)
            y =  x + np.random.normal(0,self.r)
            ys.append(y)
        return xs, ys
  
class LogisticMap():
    def __init__(self,
        length,
        dt,
        q,
        r,
        obs_dim = 1,
        ):
        self.length = length
        self.dt = dt
        self.q = q
        self.r = r
        self.obs_dim = obs_dim
    def h_fn(self, x):
        return x

    def R_inv(self, resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    
    def generate(self):
        xs=[]
        ys=[]
        chaos = np.random.uniform(2.8, 4)
        x=np.random.rand()
        time_points = np.arange(0, self.length, self.dt)
        for t in time_points:
            x = chaos*x*(1-x)
            xs.append(x)
            y =  x + np.random.normal(0,self.r)
            ys.append(y)
        xs = np.array(xs).reshape(-1, 1)
        ys = np.array(ys).reshape(-1, 1)
        return xs, ys
  
class RandomWalk():
    def __init__(self,
        length,
        dt,
        q,
        r,
        obs_dim = 1,
        ):
        self.length = length
        self.dt = dt
        self.q = q
        self.r = r
        self.obs_dim = obs_dim
    def h_fn(self, x):
        return x
    def R_inv(self, resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    def generate(self):
        xs=[]
        ys=[]
        x=np.random.rand()
        time_points = np.arange(0, self.length, self.dt)
        for t in time_points:
            x = x + np.random.normal(0,self.q)
            xs.append(x)
            y =  x + np.random.normal(0,self.r)
            ys.append(y)
        xs = np.array(xs).reshape(-1, 1)
        ys = np.array(ys).reshape(-1, 1)
        return xs, ys
  
class xDIndependentSinusoidalWaves():
    def __init__(self,
        length,
        dt,
        q,
        r,
        obs_dim = 2,
        ):
        self.length = length
        self.dt = dt
        self.q = q
        self.r = r
        self.obs_dim = obs_dim
        
    def h_fn(self, x):
        return x

    def R_inv(self, resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        # normalize per feature, not per time
        R_inv = R_inv / (R_inv.std(dim=(1, 2), keepdim=True) + 1e-5)
        return R_inv
    
    
    def generate(self):
        amplitude = np.random.uniform(0.5, 2, size=(self.obs_dim, 1))
        frequency = np.random.uniform(0.5, 2, size=(self.obs_dim, 1))
        phase = np.random.randn(self.obs_dim, 1) * 2 * np.pi
        
        t = np.arange(0, self.length, self.dt)
        
        x = np.sin(frequency * t + phase) * amplitude
        x += np.random.normal(0, self.q, size=x.shape)
    
        y = x + np.random.normal(0, self.r, size=x.shape)
        
        return x.T, y.T
 
class TwoDDependentSinusoidalWaves():
    def __init__(self,
        length,
        dt,
        q,
        r,
        obs_dim = 2,
        ):
        self.length = length
        self.dt = dt
        self.q = q
        self.r = r
        self.obs_dim = obs_dim
        
    def h_fn(self, x):
        return x

    def R_inv(self, resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        # normalize per feature, not per time
        R_inv = R_inv / (R_inv.std(dim=(1, 2), keepdim=True) + 1e-5)
        return R_inv
    
    
    def generate(self):
        f1 = np.random.uniform(0.5,2.0)
        f2 = np.random.uniform(0.5,2.0)
        a1 = np.random.uniform(0.5,2.0)
        a2 = np.random.uniform(0.5,2.0)
        
        alpha = np.random.uniform(0.5, 1.5)

        t = np.arange(0, self.length, self.dt)

        # main sinusoid
        x1 = a1 * np.sin(f1 * t)
        # quadrature sinusoid + nonlinear phase warp
        x2 = a2 * np.cos(f2 * t + alpha * x1)

        x = np.stack([x1, x2], axis=0)


        x = x + np.random.normal(0, self.q, size=x.shape)
        y = x + np.random.normal(0, self.r, size=x.shape)

        return x.T, y.T
  
class MassSpringChain:
    def __init__(self, length, dt, q, r, obs_dim=3):
        self.N = obs_dim
        self.state_dim = 2 * self.N

        self.length = length
        self.dt = dt
        self.q = q
        self.r = r
        raise NotImplementedError

    
class Lorenz():
    def __init__(self,
        length,
        dt,
        q,
        r,
        obs_dim = 3,
        ):
        self.length = length
        self.dt = dt
        self.q = q
        self.r = r
        self.obs_dim = obs_dim
    def h_fn(self, x):
        return x

    def R_inv(self, resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=(1,2), keepdim=True) + 1e-5)
        return R_inv
    
    
    def lorenz_step(self,x, sigma=10.0, rho=28.0, beta=8.0/3.0):
        dx = np.array([sigma*(x[1]-x[0]),
                    x[0]*(rho-x[2]) - x[1],
                    x[0]*x[1] - beta*x[2]])
        return dx

    def generate(self):
        """Generates one Lorenz trajectory of length self.length."""
        dt = self.dt
        q = self.q
        steps = np.arange(0,self.length,self.dt)

        traj = np.zeros((len(steps), 3), dtype=np.float32)

        x = np.random.randn(3)
        
        for i in range(len(steps)):
            # RK4 integration
            k1 = self.lorenz_step(x)
            k2 = self.lorenz_step(x + 0.5 * dt * k1)
            k3 = self.lorenz_step(x + 0.5 * dt * k2)
            k4 = self.lorenz_step(x + dt * k3)
            x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

            # Optional process noise
            if q > 0:
                x += q * np.random.randn(3)

            traj[i] = x


        # Add observation noise if specified
        obs = traj[..., :self.obs_dim]
        if self.r > 0:
            obs = obs + self.r * np.random.randn(*obs.shape)

        return traj,obs
