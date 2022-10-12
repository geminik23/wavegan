import unittest
import torch
from model import WaveGanDiscriminator, WaveGanGenerator, UpsampleConv


class WaveGanTestCase(unittest.TestCase):
    def test_tensor_independently_shift(self):
        batch = 2
        length = 3
        channels = 2
        n = 2 # shift

        # shifts = (torch.zeros((batch, )).random_(0, n*2+1)-n).long()
        # data = torch.randn((batch, channels, length))
        shifts = [1, -1]
        data = torch.tensor([[[1,2,3], [4, 5, 6]]]).repeat((2, 1, 1)) 

        # this method shift entire 1d over batch
        # data = data.roll(shifts=shifts, dims=[-1]*len(shifts))

        ## use the gather
        indices = torch.arange(length).view((1, 1, length)).repeat((batch, channels, 1))
        for i in range(batch):
            indices[i] = (indices[i] - shifts[i]) % length
        
        y = data.gather(2, indices)

        target = torch.tensor([
            [[3, 1, 2], [6, 4, 5]],
            [[2, 3, 1], [5, 6, 4]]
        ])
        self.assertTrue(torch.equal(y, target))


    def test_discriminator(self):
        batch_size = 32
        input_size = (64*64*4)
        in_channels = 1

        target_shape = torch.Size([batch_size, 1])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = torch.randn((batch_size, in_channels, input_size))
        D = WaveGanDiscriminator(in_channels)
        x = x.to(device)
        D = D.to(device)

        y = D(x)

        self.assertEqual(y.shape, target_shape)

    def test_upsample(self):
        ch = 16
        length = 16
        up = UpsampleConv(ch, ch//2, 25, 4)
        x = torch.randn((32, ch, length))
        y = up(x)

        target_shape = torch.Size([32, ch//2, length*4])
        self.assertEqual(y.shape, target_shape)

    def test_generator(self):
        batch_size = 32
        latent_size = 100
        
        target_length = (64*64*4)

        G = WaveGanGenerator(latent_size)
        x = torch.randn((batch_size, latent_size))
        y = G(x)

        self.assertEqual(y.shape, torch.Size([batch_size, 1, target_length])) # batch_size, out_channels, lengths




if __name__ == '__main__':
    unittest.main(verbosity=2)