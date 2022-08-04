import torch
#from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
import sidechainnet as scn
from helper import init_loss_optimizer
import matplotlib.pyplot as plt

def validation(model, datasplit, device, loss_fn, mode):
    """Evaluate a model (sequence->sin/cos represented angles [-1,1]) on MSE."""
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in datasplit:
            # Prepare variables and create a mask of missing angles (padded with zeros)
            # The mask is repeated in the last dimension to match the sin/cos represenation.
            if mode == 'seqs':
                seqs = batch.int_seqs.to(device).long()
            elif mode == 'pssms':
                seqs = batch.seq_evo_sec.to(device)
            mask_ = batch.msks.to(device)
            true_angles_sincosine = scn.structure.trig_transform(batch.angs).to(device)
            mask = (batch.angs.ne(0)).unsqueeze(-1).repeat(1, 1, 1, 2)

            # Make predictions and optimize
            predicted_angles = model(seqs, mask = mask_)
            loss = loss_fn(predicted_angles[mask], true_angles_sincosine[mask])
            
            total += loss
            n += 1

    return torch.sqrt(total/n)

def train(model, config, dataloader, device):
    optimizer, batch_losses, epoch_training_losses, epoch_validation10_losses, epoch_validation90_losses, mse_loss = init_loss_optimizer(model, config)
    for epoch in range(config.epoch):
        print(f'Epoch {epoch}')
        progress_bar = tqdm(total=len(dataloader['train']), smoothing=0)
        for batch in dataloader['train']:
            # Prepare variables and create a mask of missing angles (padded with zeros)
            # Note the mask is repeated in the last dimension to match the sin/cos represenation.
            if config.mode == 'seqs':
                seqs = batch.int_seqs.to(device).long()
            elif config.mode == 'pssms':
                seqs = batch.seq_evo_sec.to(device)
            mask_ = batch.msks.to(device)
            true_angles_sincos = scn.structure.trig_transform(batch.angs).to(device)
            mask = (batch.angs.ne(0)).unsqueeze(-1).repeat(1, 1, 1, 2)

            # Make predictions and optimize
            predicted_angles = model(seqs, mask = mask_)
            loss = mse_loss(predicted_angles[mask], true_angles_sincos[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()

            # Housekeeping
            batch_losses.append(float(loss))
            progress_bar.update(1)
            progress_bar.set_description(f"\rRMSE Loss = {np.sqrt(float(loss)):.4f}")
        # Evaluate the model's performance on train-eval, downsampled for efficiency
        epoch_training_losses.append(validation(model, 
                                                dataloader['train-eval'], 
                                                device,
                                                mse_loss, 
                                                config.mode))
        # Evaluate the model's performance on various validation sets
        epoch_validation10_losses.append(validation(model, 
                                        dataloader['valid-10'],
                                        device, 
                                        mse_loss,
                                        config.mode))
        epoch_validation90_losses.append(validation(model, 
                                        dataloader['valid-90'], 
                                        device, 
                                        mse_loss,
                                        config.mode))
        print(f"     Train-eval loss = {epoch_training_losses[-1]:.4f}")
        print(f"     Valid-10   loss = {epoch_validation10_losses[-1]:.4f}")
        print(f"     Valid-90   loss = {epoch_validation90_losses[-1]:.4f}")
    # Finally, evaluate the model on the test set
    print(f"Test loss = {validation(model, dataloader['test'], device, mse_loss, config.mode):.4f}")
    # Plot the loss of each batch over time
    plt.plot(np.sqrt(np.asarray(batch_losses)), label='batch loss')
    plt.ylabel("RMSE")
    plt.xlabel("Step")
    plt.title("Training Loss over Time")
    plt.savefig('TrainingLoss.png')

    # While the above plot demonstrates each batch's loss during training,
    # the plot below shows the performance of the model on several data splits
    # at the *end* of each epoch.
    plt.plot([x.cpu().detach().numpy() for x in epoch_training_losses], label='train-eval')
    plt.plot([x.cpu().detach().numpy() for x in epoch_validation10_losses], label='valid-10')
    plt.plot([x.cpu().detach().numpy() for x in epoch_validation90_losses], label='valid-90')
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    plt.title("Training and Validation Losses over Time")
    plt.legend()
    plt.savefig('ValidationLoss.png')
    
    return model