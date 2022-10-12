import torch
import torch.nn as nn


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, output_padding=1, transconv = True, upsample_factor=None):
        super().__init__()
        # maybe later nearest-upsampling
        # only transposed conv
        if transconv:
            self.conv1dtrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, input):
        return self.conv1dtrans(input)



# 'Phase shuffle randomly perturbs the phase of each layer’s activations by −n to n samples before input to the next layer'
#  'by Uniform ∼ [−n, n] samples, filling in the missing samples (dashed outlines) by reflection.' from paper
class PhaseShuffle(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, input):
        # input
        if self.n == 0: return input
        b = input.size(0)
        # shift_n -> len([-n ~ n]) = (2n+1) 
        shifts = (torch.zeros((b, )).random_(0, self.n*2+1)-self.n).long()
        
        indices = torch.arange(input.size(-1)).view((1, 1, input.size(-1))).repeat((input.size(0), input.size(1), 1))
        for i in range(b):
            indices[i] = (indices[i] - shifts[i]) % input.size(-1)
        indices = indices.to(input.device)
        return input.gather(2, indices)


