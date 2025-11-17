# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from dataGeneration import SinusoidalWaves,Lorenz,DualSinusoidalWaves,LogisticMap,RandomWalk,xDIndependentSinusoidalWaves,TwoDDependentSinusoidalWaves, MassSpringChain
from dataRetrieve import Hurdat
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import os

def get_custom_dataset(dataset_name, samples=10, context_length=80,prediction_length=20, dt=1,q=1,r=1,observation_dim=1,plot=False):
    generatingClasses = {
        "sinusoidal": SinusoidalWaves,
        "lorenz":Lorenz,
        "dualsinusoidal": DualSinusoidalWaves,
        "logistic":LogisticMap,
        "random":RandomWalk,
        "2dsinindependent":xDIndependentSinusoidalWaves,
        "xdsinindependent":xDIndependentSinusoidalWaves,
        "2dsindependent": TwoDDependentSinusoidalWaves,
        "massspringchain":MassSpringChain,

    }
    generator = generatingClasses[dataset_name](context_length+prediction_length,dt,q,r,observation_dim)

    states = []
    observations = []
    for sample in range(samples):
        state, obs = generator.generate()
        states.append(state)
        observations.append(obs)

    state_array = np.array(states)
    observation_array = np.array(observations)



    if plot:
        index = random.randint(0, len(states) - 1)
        true_state = np.array(states[index])
        noisy_state = np.array(observations[index])
        dataRange = np.arange(0, context_length + prediction_length, dt)
        dim = true_state.shape[1] if true_state.ndim > 1 else 1
        


        if dim == 1:
            fig = plt.figure(figsize=(10, 6))
            plt.title(f'{dataset_name.capitalize()} Wave with Noisy Observations')
            plt.plot(dataRange, true_state, label='True Value')
            plt.plot(dataRange, noisy_state, label='Noisy Value')
            plt.axvline(x=context_length, linestyle=':', color='r', label='End of Context')
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.show()

        elif dim == 2:
            fig = plt.figure(figsize = (10,6))
            ax1 = fig.add_subplot(121)
            ax1.set_title(f'{dataset_name.capitalize()} (2D) Dim 1')
            ax1.plot(dataRange, true_state[:, 0], label='True dim 1')
            ax1.plot(dataRange, noisy_state[:, 0], '--', label='Noisy dim 1')
            ax1.axvline(x=context_length, linestyle=':', color='r', label='End of Context')
            
            ax2 = fig.add_subplot(122)
            ax2.set_title(f'{dataset_name.capitalize()} (2D) Dim 2')
            ax2.plot(dataRange, true_state[:, 1], label='True dim 2')
            ax2.plot(dataRange, noisy_state[:, 1], '--', label='Noisy dim 2')
            ax2.axvline(x=context_length, linestyle=':', color='r', label='End of Context')
            
            plt.legend()
            plt.show()

        elif dim == 3:
            fig = plt.figure(figsize=(10, 10))
            gs = GridSpec(3, 3, figure=fig)  # 3 rows, 3 columns

            # 3D trajectory spanning the entire top row
            ax = fig.add_subplot(gs[0, :], projection='3d')
            ax.set_title(f'{dataset_name.capitalize()} (3D Trajectory)')
            ax.plot(true_state[:, 0], true_state[:, 1], true_state[:, 2], label='True Trajectory')
            ax.plot(noisy_state[:, 0], noisy_state[:, 1], noisy_state[:, 2], linestyle='--', label='Noisy Trajectory')
            ax.legend()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            # Bottom row: one, two, three dims
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set_title('First Dimension')
            ax2.plot(dataRange, true_state[:, 0], label='True')
            ax2.plot(dataRange, noisy_state[:, 0], label='Noisy')
            ax2.axvline(x=context_length, linestyle=':', color='r', label='End of Context')
            ax2.legend()

            ax3 = fig.add_subplot(gs[1, 1])
            ax3.set_title('Second Dimension')
            ax3.plot(dataRange, true_state[:, 1], label='True')
            ax3.plot(dataRange, noisy_state[:, 1], label='Noisy')
            ax3.axvline(x=context_length, linestyle=':', color='r', label='End of Context')
            ax3.legend()

            ax4 = fig.add_subplot(gs[1, 2])
            ax4.set_title('Third Dimension')
            ax4.plot(dataRange, true_state[:, 2], label='True')
            ax4.plot(dataRange, noisy_state[:, 2], label='Noisy')
            ax4.axvline(x=context_length, linestyle=':', color='r', label='End of Context')
            ax4.legend()

            plt.tight_layout()
            plt.show()
            
        else:
            print(f"Unsupported dim {dim}")
            raise ValueError



    custom_data = [
        {
            "state": np.array(state),         # shape (seq_len, 1)
            "observation": np.array(obs),    # shape (seq_len, 1)
        }
        for obs, state in zip(observation_array, state_array)
    ]
    
    
    
    custom_data = np.array(custom_data)

    return custom_data, generator



def get_stored_dataset(dataset_name, config=None,length=5,plot=False):
    retrievers = {
        "hurricane": Hurdat,
    }

    retriever = retrievers[dataset_name](length=length,plot=plot)
    
    observations = retriever.generate()

    custom_data = [
        {
            "state":np.array(obs),#only needed for batching ig,
            "observation":np.array(obs),
        }
        for obs in observations
    ]
    custom_data = np.array(custom_data)
    
    config['dt'] = retriever.dt
    config['r']=retriever.r
    config['q']=retriever.q
    config['observation_dim']=retriever.obs_dim
    config['data_samples']=len(observations)

    

    return custom_data,retriever,config
    