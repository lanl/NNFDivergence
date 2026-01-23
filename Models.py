import torch
import torch.nn as nn
import numpy as np 
import multiprocessing as mp
from scipy.sparse import csr_matrix 
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree

class CNN(nn.Module):
    def __init__(self, num_classes: int, dropout_prob: float = 0.0, data_type: str = 'PDS_ChemCam'):
        super().__init__()

        layers = [
            nn.Conv1d(1, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.MaxPool1d(kernel_size=5))

        layers += [
            nn.Conv1d(32, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.MaxPool1d(kernel_size=5))

        layers += [
            nn.Conv1d(16, 8, kernel_size=5),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.MaxPool1d(kernel_size=5))

        self.cnn = nn.Sequential(*layers)

        # depending on the data type, set the final output dimension
        if data_type == 'PDS_ChemCam':
            self.fc = nn.Linear(376, num_classes)
        elif data_type == 'PDS_SuperCam':
            self.fc = nn.Linear(496, num_classes)
    def forward(self, x):
        """
        x: (batch, 1, L) where L can be any length (as long as it survives pooling)
        """
        feat_map = self.cnn(x)          # (B, 8, L')
        flattern_feat = feat_map.view(feat_map.size(0), -1)  # (B, 8, 1)
        logits = self.fc(flattern_feat)          # (B, num_classes)
        return feat_map, logits



class TwoSampletrain():
    def __init__(self, args, predmodel, device, train_loader, val_loader, pred_optimizer):
        self.predmodel = predmodel
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pred_optimizer = pred_optimizer
        self.TSloss = 0 
        self.test_TSLoss = 0 
        self.args = args
        self.Lambda = args.Lambda
        self.weight = args.weight
        self.gamma = args.gamma
    
    def TwoSampleLoss(self, Pred, Targets): # Multi-variate differentiable f-divergence
        SumDic = {}
        TwoSamples = torch.cat((Pred, Targets)).view(-1, Targets.shape[-1])
        ZeroLabels = np.zeros((len(Pred), 1))
        OneLabels = np.ones((len(Targets), 1))
        Labels = np.vstack((ZeroLabels, OneLabels))
        if self.device.type == "mps":
        # compute on CPU and move result back
            PairWiseDist=-torch.cdist(TwoSamples.to("cpu"), TwoSamples.to("cpu")).to(self.device)
        else:
            PairWiseDist = -torch.cdist(TwoSamples, TwoSamples)/self.Lambda
        ExpPairWiseDist = torch.exp(PairWiseDist)
        Threshold = torch.tensor(1e-6).to(self.device) # used to thresholding the denominator of the cut-edge ratio to prevent NaN results.

        # sum of the distance
        L = len(TwoSamples)
        for i in range(L):
            SumDic[i] = torch.sum(ExpPairWiseDist[i]) - ExpPairWiseDist[i, i]
        # compute loss
        SoftCutEdgeNum = None 
        for i in range(len(TwoSamples)): # differentiable f-divergence
            # print(i, SumDic[i])
            for j in range(len(TwoSamples)):
                if j != i and Labels[i] != Labels[j]:
                    if SoftCutEdgeNum == None:
                        SoftCutEdgeNum = torch.divide(ExpPairWiseDist[i][j],  torch.maximum(SumDic[i], Threshold)) # add 1e-6 to prevent nan results.
                    else:
                        SoftCutEdgeNum = torch.add(torch.divide(ExpPairWiseDist[i][j],  torch.maximum(SumDic[i], Threshold)), SoftCutEdgeNum)
        Loss = 1 - 2*SoftCutEdgeNum / L # Dp divergence =(1 - cutedge proportion) * 2 - 1
        return Loss 
   
    def test(self):
        self.predmodel.eval(); Count = 0
        self.test_TSLoss = 0;
        criterion = nn.MSELoss()
        for batch_idx, (Feats, targets) in enumerate(self.val_loader):
            Feats, targets = Feats.to(self.device), targets.to(self.device)

            """
            Train the prediction network 
            """
            Pred_out,  Pred_out2 = self.predmodel(Feats)
            squeeze_Pred_out2 = Pred_out2.squeeze()

            TSLoss = self.TwoSampleLoss(Pred_out2, targets)
            MSEloss = criterion(squeeze_Pred_out2, targets)

            self.test_TSLoss+=(self.weight * TSLoss.item() + MSEloss.item())
            Count+=1
        self.test_TSLoss = self.test_TSLoss/ Count
        # print(self.test_TSLoss)
    def train(self, epoch):
        self.predmodel.train()
        criterion = nn.MSELoss()
        for batch_idx, (Feats, targets) in enumerate(self.train_loader):
            Feats, targets = Feats.to(self.device), targets.to(self.device)
            self.pred_optimizer.zero_grad()
           
            """
            Train the prediction network to reduce the differentiable two-sample statistic   
            """
            Pred_out,  Pred_out2 = self.predmodel(Feats)

            TSloss = self.TwoSampleLoss(Pred_out2, targets)

            MSEloss = criterion(Pred_out2, targets)

            TotalLoss = MSEloss + self.weight*(TSloss - self.gamma)**2
            if self.args.LossType2 == 'L1_fdiv':
                l1_parameters = []
                for parameter in self.predmodel.parameters():
                    l1_parameters.append(parameter.view(-1))
                l1 = self.args.l1delta * torch.abs(torch.cat(l1_parameters)).sum()
                TotalLoss+=l1 
            
            TotalLoss.backward()
            self.pred_optimizer.step()
            self.TSloss = TotalLoss.item()


def FDivergence(X, Y):
    """
    Fridman Rafsky statstic (graph estimate of a f-divergence)
    """

    """
    Construct minimum spanning tree
    """
    NumCores = mp.cpu_count();
    Sample0 = np.hstack((X, np.zeros((len(X), 1)))); Sample1 = np.hstack((Y, np.ones((len(Y), 1))))
    Data = np.vstack((Sample0,Sample1)); GraphMatrix = pairwise_distances(Data[:, :-1], n_jobs = NumCores)
    for i in range(len(GraphMatrix)):
         for j in range(i + 1, len(GraphMatrix)):
              if GraphMatrix[i, j] == 0:
                   GraphMatrix[i,j] = 1e-10 # assign small edge weight for repetitive data

    for i in range(len(GraphMatrix)):
        GraphMatrix[i, :i + 1] = 0
    Tcsr = minimum_spanning_tree(csr_matrix(GraphMatrix))	
    Tree = Tcsr.toarray()
    for i in range(len(Tree)):
        Tree[np.where(Tree[i] != 0), i] = Tree[i][np.where(Tree[i] != 0)]
    BaseG = Tree.astype(bool).astype(int); 

    """
    Cut-edge count
    """
    R = 0; G = 0; n = len(X); m = len(Y); N = m + n; 
    for i in range(len(BaseG)):
        for j in range(i + 1, len(BaseG)):
            if BaseG[i, j] == 1:
                G+=1
                if Data[i, -1] != Data[j, -1]:
                    R+=1
    Dp = 1 - 2 * R /N
 
    return max(Dp, 0)

def train(args, model, device, train_loader, optimizer):
    model.train()

    # MSE
    criterion = nn.MSELoss()

    for batch_idx, (Feats, targets) in enumerate(train_loader):
        Feats, targets = Feats.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs1,  outputs2 = model(Feats)
        # outputs2 = outputs2.squeeze()
        loss = criterion(outputs2, targets)
        if args.LossType == 'L1':
            l1_parameters = []
            for parameter in model.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = args.l1delta * torch.abs(torch.cat(l1_parameters)).sum()
            loss+=l1           
       
        loss.backward()
        optimizer.step()
    return loss.item()

def test(args, model, device, test_loader, scaler):
    model.eval()

    Pred = np.zeros((0, test_loader.dataset.Labels.shape[-1]))
    TrueLabels = np.zeros((0, test_loader.dataset.Labels.shape[-1]));
      
    with torch.no_grad():
        for (Feats, targets) in test_loader:
            Feats, targets = Feats.to(device), targets.to(device)
            Output1, Output2 = model(Feats)
            # Output2 = Output2.squeeze()
            Output2 = Output2.reshape((-1, Output2.shape[-1]))
            Pred = np.vstack((Pred, Output2.cpu().detach().numpy()))
            TrueLabels = np.vstack((TrueLabels, targets.cpu().detach().numpy()))

    Pred = Pred * scaler.scale_ + scaler.mean_
        
    Err1 = mean_squared_error(Pred, TrueLabels)

    Err2 = FDivergence(Pred, TrueLabels)
    return Pred, Err1, Err2
    
# Create a dataset class for the One-data training
class MyAIDataset(Dataset):
    def __init__(self, Feats, Labels):
        self.FeatLen = Feats.shape[-1]
        self.Feats = np.float32(Feats).reshape((-1, 1, self.FeatLen))
        self.Labels = np.float32(Labels) 

    def __len__(self):
        return len(self.Feats)

    def __getitem__(self, idx):
        return self.Feats[idx], self.Labels[idx]
