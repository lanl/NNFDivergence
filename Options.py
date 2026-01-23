import argparse

# for all approaches 
parser = argparse.ArgumentParser(description='Advanced AI model training for the spectral data')

parser.add_argument('--DataType', type=str, default='PDS_SuperCam', help='PDS_ChemCam or PDS_SuperCam')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--oxide_id', type=int, default=0,
                    help='for the single-oxide modelling; index of available oxides')

# PLS hyper-parameters 
parser.add_argument('--EmbeddingDim', type=int, default=8, 
                    help='embedding dimension for pls')

# neural network config. 
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--TrialNum', type=int, default=1, metavar='N',
                    help='Trial number to average results')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--Lambda', type=float, default=1, metavar='M',
                    help='scale for the distance')
parser.add_argument('--weight', type=float, default=0.0001, metavar='M',
                    help='balancing factor')
parser.add_argument('--gamma', type=float, default=0, metavar='M',
                    help='regularization strengh')
parser.add_argument('--l2delta', type=float, default=0.03, metavar='M',
                    help='regularizer strength for l2')

parser.add_argument('--l1delta', type=float, default=0.03, metavar='M',
                    help='regularizer strength for l1')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')

parser.add_argument('--save-model', type=int, default=1,
                    help='whether to save model')

parser.add_argument('--Dropout_prob', type=float, default=0,
                    help='whether to use dropout')

parser.add_argument('--LossType', type=str, default='L2', help='L1, L2 or Dropout')

parser.add_argument('--LossType2', type=str, default='L1_fdiv', help='fdiv, L2_fdiv, L1_fdiv or Dropout_fdiv')


