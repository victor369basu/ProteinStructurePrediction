import sidechainnet as scn
import torch
from sidechainnet.structure.structure import inverse_trig_transform
import py3Dmol

def build_visualizable_structures(model, data, config, device):
    """Build visualizable structures for one batch of model's predictions on data."""
    with torch.no_grad():
        for batch in data:
            if config.mode == "seqs":
                model_input = batch.int_seqs.to(device)
            elif config.mode == "pssms":
                model_input = batch.seq_evo_sec.to(device)
            mask_ = batch.msks.to(device)
            # Make predictions for angles, and construct 3D atomic coordinates
            predicted_angles_sincos = model(model_input, mask = mask_)
            # Because the model predicts sin/cos values, we use this function to recover the original angles
            predicted_angles = inverse_trig_transform(predicted_angles_sincos)

            # EXAMPLE
            # Use BatchedStructureBuilder to build an entire batch of structures
            sb_pred = scn.BatchedStructureBuilder(batch.int_seqs, predicted_angles.cpu())
            sb_true = scn.BatchedStructureBuilder(batch.int_seqs, batch.crds.cpu())
            break
    return sb_pred, sb_true

def plot_protein(exp1, exp2):
    p = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', viewergrid=(2,1))
    p.addModel(open(exp1,'r').read(),'pdb', viewer=(0,0))
    p.addModel(open(exp2,'r').read(),'pdb', viewer=(1,0))
    p.setStyle({'cartoon': {'color':'spectrum'}})
    p.zoomTo()
    p.show()