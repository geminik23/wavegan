import torch.nn as nn
from modules import PhaseShuffle, UpsampleConv
import torch.nn.functional as F
import math

class WaveGanGenerator(nn.Module):
    def __init__(self, latent_size=100, model_size=64, out_channels= 1, kernel_size=25, stride=4, upsample_factor=4):
        super().__init__()

        self.latent_size = latent_size
        self.model_size = model_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample_factor = upsample_factor

        self.start_dim = 256*self.model_size
        self.start_channels = 16*self.model_size

        len_upconvs = 4
        self.latent_net = nn.Linear(latent_size, self.start_dim)
        convs = []
        for i in range(len_upconvs):
            convs.append(UpsampleConv(self.start_channels//(2**i), self.start_channels//(2**(i+1)), kernel_size, stride))
            convs.append(nn.ReLU())
        self.upsample_convs = nn.Sequential( *convs)
        self.final_conv = UpsampleConv(model_size, out_channels, kernel_size, stride)
        self.tanh = nn.Tanh()
        
        
    def forward(self, input):
        # input : [B, latent_dim]
        x = self.latent_net(input).view(-1, self.start_channels, 16)
        x = F.relu(x)
        x = self.upsample_convs(x)
        # F.tanh is deprecated? use torch.tanh or nn.Tanh
        return self.tanh(self.final_conv(x))  # [B, C, 64*64*4]



class WaveGanDiscriminator(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, model_size=64, kernel_size=25, stride=4, phase_shift=2, leaky=0.2):
        super().__init__()

        self.in_channels = in_channels
        self.model_size = model_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.leaky_rate = leaky
        self.phase_shift = phase_shift
        self.num_classes = num_classes

        self.input_size = 64*64*4
        
        convs = [
            nn.Conv1d(in_channels, model_size, 25, stride=4, padding=11),
            nn.LeakyReLU(leaky)
        ]

        len_conv = int(math.log(self.input_size/2, 4) - 1)
        for i in range(len_conv-1):
            convs += [
                nn.Conv1d(model_size * 2**i, model_size * 2**(i+1), 25, stride=4, padding=11),
                nn.LeakyReLU(leaky),
            ]
            if i+1 != len_conv-1:
                convs += [PhaseShuffle(self.phase_shift)]
            pass


        self.convs = nn.Sequential(*convs)
        self.fc = nn.Linear(256 * model_size, 1)

    def forward(self, input, condition=None):
        x = self.convs(input)
        x = x.view(-1, 256 * self.model_size)
        return self.fc(x)


