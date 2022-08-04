import torch
import sidechainnet as scn
import random
import os
import numpy as np

from visualize import build_visualizable_structures, plot_protein
from model import ProteinNet
from trainer import train
from config import get_parameters

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# To train with a GPU, go to Runtime > Change runtime type
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device} for training.")
def main(config, dataloader):
    print("Available Dataloaders =", list(dataloader.keys()))

    # Create the model and move it to the GPU
    model = ProteinNet(d_hidden=config.d_hidden,
                            dim=config.dim,
                            d_in=config.d_in,
                            d_embedding=config.d_embedding,
                            heads = config.n_heads,
                            dim_head = config.head_dim,
                            integer_sequence=config.integer_sequence)
    model = model.to(device)

    trained_model = train(model, config, dataloader, device)
    if os.path.exists(config.model_save_path)==False:
        os.mkdir(config.model_save_path)
    torch.save(trained_model.state_dict(), '{}/model_weights.pth'.format(config.model_save_path))

def plot(idx, dataloader, config):
    model =  ProteinNet(d_hidden=config.d_hidden,
                        dim=config.dim,
                        d_in=config.d_in,
                        d_embedding=config.d_embedding,
                        heads = config.n_heads,
                        dim_head = config.head_dim,
                        integer_sequence=config.integer_sequence)
    model = model.to(device)
    model.load_state_dict(torch.load('{}/model_weights.pth'.format(config.model_save_path)))

    if os.path.exists('./plots')==False:
        os.mkdir('./plots')
    s_pred, s_true = build_visualizable_structures(model, dataloader['train'], config, device)
    s_pred.to_pdb(idx,path='./plots/{}_pred.pdb'.format(idx))
    s_true.to_pdb(idx,path='./plots/{}_true.pdb'.format(idx))
    plot_protein('./plots/{}_pred.pdb'.format(idx), './plots/{}_true.pdb'.format(idx))

if __name__ == '__main__':
    config = get_parameters()
    print("Model Configuration: ")
    print(config)
    # Load the data in the appropriate format for training.
    dataloader = scn.load(
                with_pytorch="dataloaders",
                batch_size=config.batch, 
                dynamic_batching=False)
    if config.train:
        main(config, dataloader)
    else:
        plot(config.idx, dataloader, config)