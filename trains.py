import torch
import numpy as np
from tqdm.autonotebook import tqdm

def train_wgan_gp(cp_file_template, G, D, latent_size, optimizer_G, optimizer_D, data_loader, epochs, start_epoch=1, device='cpu', cp_interval=5):
    """
    Args:
    cp_file_template : save filename template e.g "model_{}.pt"

    Return:
    tuple: (generator loss, discriminator loss)
    """

    G.to(device)
    D.to(device)

    g_losses = []
    d_losses = []
 
    G.train()
    D.train()
    last_epoch = start_epoch - 1
    for epoch in tqdm(range(start_epoch, epochs+start_epoch)):
        _glosses = []
        _dlosses = []
        for data in tqdm(data_loader, leave=False):
            if isinstance(data, tuple) or isinstance(data, list):
                data, _ = data

            real = data.to(device)

            batch_size = real.size(0)

            ### Step 1 : update D
            D.zero_grad()
            G.zero_grad()

            # real
            d_real = D(real)

            # fake
            z = torch.randn(batch_size, latent_size, device=device)
            fake = G(z) 
            d_fake = D(fake)

            # gradient penalty
            eps_shape = [batch_size]+[1]*(len(data.shape)-1)
            eps = torch.rand(eps_shape, device=device)
            fake = eps*real + (1-eps)*fake
            output = D(fake) 

            grad = torch.autograd.grad(outputs=output, inputs=fake,
                                  grad_outputs=torch.ones(output.size(), device=device),
                                  create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
            d_grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
            ##

            errd = (d_fake-d_real).mean() + d_grad_penalty.mean()*10
            errd.backward()
            optimizer_D.step()
            
            _dlosses.append(errd.item())

            #### Step 2 : update G
            D.zero_grad()
            G.zero_grad()

            noise = torch.randn(batch_size, latent_size, device=device)
            output = -D(G(noise))
            errg = output.mean()
            errg.backward()
            optimizer_G.step()
            
            _glosses.append(errg.item())

        g_losses.append(np.mean(_glosses))
        d_losses.append(np.mean(_dlosses))

        last_epoch = epoch

        if epoch % cp_interval == 0:
            # save check_points
            torch.save({
                'losses': g_losses,
                'epoch': epoch,
                'model_state_dict': G.state_dict(), 
                }, cp_file_template.format('g', epoch))

            torch.save({
                'losses': d_losses,
                'epoch': epoch,
                'model_state_dict': D.state_dict(), 
            }, cp_file_template.format('d', epoch))
        pass

    # save check_points
    torch.save({
        'losses': g_losses,
        'epoch': epoch,
        'model_state_dict': G.state_dict(), 
        }, cp_file_template.format('g', last_epoch))

    torch.save({
        'losses': d_losses,
        'epoch': epoch,
        'model_state_dict': D.state_dict(), 
    }, cp_file_template.format('d', last_epoch))

    return g_losses, d_losses

