import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model Hyper-parameters
    parser.add_argument('--d_in', dest='d_in', default=49, type=int,
                        help="Model input dimension.")
    parser.add_argument('--d_hidden', dest='d_hidden', default=512, type=int,
                        help="Dimensionality of RNN hidden state.")
    parser.add_argument('--dim', dest='dim', default=256, type=int,
                        help="Attention Layer Dim.")
    parser.add_argument('--d_embedding', dest='d_embedding', default=32, type=int,
                        help="Embedding dimension.")
    parser.add_argument('--n_heads', dest='n_heads', default=8, type=int,
                        help="Number of heads in Attention Layer.")
    parser.add_argument('-h_dim','--head_dim', dest='head_dim', default=64, type=int,
                        help="Dimension of heads in Attention Layer.")
    parser.add_argument('-int_seq','--integer_sequence', dest='integer_sequence', type=str2bool, default=False,
                        help="Dimension of heads in Attention Layer.")

    # Training parameters
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', default=0.001, type=float,
                        help="Learning Rate.")
    parser.add_argument('-e', '--epoch', type=int, default=10, help="Training Epochs.")
    parser.add_argument('-b', '--batch', dest='batch',type=int, default=4,
                        help="Batch size during each training step.")
    parser.add_argument('-t','--train', type=str2bool, default=False,help="True when train the model, \
                        else used for testing.")
    parser.add_argument('--mode', type=str, default='pssms', choices=['pssms', 'seqs'],
                        help="Mode of trainig the model. Select the input of model either to be PSSM-Position Specific Scoring Matrix \
                              or Seqs(Protein Sequence)")
    # Validation
    parser.add_argument('--idx', type=int, default=0,
                        help="Validation index")

    # Base Directory
    parser.add_argument('-m', '--model_save_path', type=str, default='./models',
                        help="Path to Saved model directory.")
    return parser.parse_args()
    
