import torch
from torch import nn
from attention_layer import Attention
import sidechainnet as scn

class ProteinNet(nn.Module):
    """A protein sequence-to-angle model that consumes integer-coded sequences."""
    def __init__(self,
                 d_hidden,
                 dim,
                 d_in=21,
                 d_embedding=32,
                 heads = 8,
                 dim_head = 64,
                 integer_sequence=True,
                 n_angles=scn.structure.build_info.NUM_ANGLES):
        
        super(ProteinNet, self).__init__()
        # Dimensionality of RNN hidden state
        self.d_hidden = d_hidden
      
        self.attn = Attention(dim = dim, 
                                heads = heads,
                                dim_head = dim_head)
        # Output vector dimensionality (per amino acid)
        self.d_out = n_angles * 2
        # Output projection layer. (from RNN -> target tensor)
        self.hidden2out = nn.Sequential(
                            nn.Linear(d_embedding, d_hidden),
                            nn.GELU(),
                            nn.Linear(d_hidden, self.d_out)
                                    )
        self.out2attn = nn.Linear(self.d_out, dim)
        self.final = nn.Sequential(
                            nn.GELU(),
                            nn.Linear(dim, self.d_out))
        self.norm_0 = nn.LayerNorm([dim])
        self.norm_1 = nn.LayerNorm([dim])
        self.activation_0 = nn.GELU()
        self.activation_1 = nn.GELU()

        # Activation function for the output values (bounds values to [-1, 1])                                  
        self.output_activation = torch.nn.Tanh()

        # We embed our model's input differently depending on the type of input
        self.integer_sequence = integer_sequence
        if self.integer_sequence:
            self.input_embedding = torch.nn.Embedding(d_in, d_embedding, padding_idx=20)
        else:
            self.input_embedding = torch.nn.Linear(d_in, d_embedding)
    def get_lengths(self, sequence):
        """Compute the lengths of each sequence in the batch."""
        if self.integer_sequence:
            lengths = sequence.shape[-1] - (sequence == 20).sum(axis=1)
        else:
            lengths = sequence.shape[1] - (sequence == 0).all(axis=-1).sum(axis=1)
        return lengths.cpu()

    def forward(self, sequence, mask=None):
        """Run one forward step of the model."""
        # First, we compute sequence lengths
        lengths = self.get_lengths(sequence)

        # Next, we embed our input tensors for input to the RNN
        sequence = self.input_embedding(sequence)

        # Then we pass in our data into the RNN via PyTorch's pack_padded_sequences
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                         lengths,
                                                         batch_first=True,
                                                         enforce_sorted=False)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(sequence,
                                                                      batch_first=True)
        # At this point, output has the same dimentionality as the RNN's hidden
        # state: i.e. (batch, length, d_hidden). 
      
        # We use a linear transformation to transform our output tensor into the
        # correct dimensionality (batch, length, 24)
        output = self.hidden2out(output)
        output = self.out2attn(output)
        output = self.activation_0(output)
        output = self.norm_0(output)
        output = self.attn(output, mask=mask)
        output = self.activation_1(output)
        output = self.norm_1(output)
        output = self.final(output)
      
        # Next, we need to bound the output values between [-1, 1]
        output = self.output_activation(output)

        # Finally, reshape the output to be (batch, length, angle, (sin/cos val))
        output = output.view(output.shape[0], output.shape[1], 12, 2)

        return output