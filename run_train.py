import os
import numpy as np
import torch
from torch.optim import Adam

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
from datasets import get_speechcommand_dataset
from config import Config
from model import WaveGanDiscriminator, WaveGanGenerator
from trains import train_wgan_gp


##
# Hyperparameters from config
config = Config()

result_dir = 'results'
os.makedirs(result_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)

##
# Dataset
sc_train, _ = get_speechcommand_dataset(config, config.audio_length)
train_loader = DataLoader(sc_train, batch_size=config.batch_size, pin_memory=True, shuffle=True, num_workers=config.num_workers)


##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


##
# Ready for the model and optimizers
G = WaveGanGenerator(config.latent_size, config.model_size, out_channels=config.audio_channel)
D = WaveGanDiscriminator(config.audio_channel, config.model_size, phase_shift=config.phase_shift_factor)

optimizer_g = Adam(G.parameters(), lr=config.g_lr, betas=config.adam_betas)
optimizer_d = Adam(D.parameters(), lr=config.d_lr, betas=config.adam_betas)


##
# Train the models
g_losses, d_losses = train_wgan_gp(os.path.join(config.checkpoint_dir, config.checkpoint_file_template), 
                          G, D, config.latent_size, optimizer_g, optimizer_d, train_loader,
                          epochs=config.epochs, cp_interval=config.checkpoint_interval, device=device)

## 
# Plot the losses
plt.figure(figsize=(10,5))
plt.title('Conditional WaveGAN Generator and Discriminator Losses')
plt.plot(np.convolve(g_losses, np.ones((100,))/100, mode='valid') ,label='G') 
plt.plot(np.convolve(d_losses, np.ones((100,))/100, mode='valid') ,label='D')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
