import torch
import torch.nn as nn
import numpy as np 
import multiprocessing as mp
from scipy.sparse import csr_matrix 
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree

class ChemCam_CNN(nn.Module):
    def __init__(self, dropout_prob=0):
        super(ChemCam_CNN, self).__init__()
        if dropout_prob > 0:
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 32, 5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.MaxPool1d(5),

                nn.Conv1d(32, 16, 5),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.MaxPool1d(5),

                nn.Conv1d(16, 8, 5),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.MaxPool1d(5)
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 32, 5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(5),

                nn.Conv1d(32, 16, 5),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(5),

                nn.Conv1d(16, 8, 5),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(5)
            )
        self.fc =  nn.Linear(376, 10)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out2 = self.fc(out)
        return out, out2
    
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
        self.delta = args.delta
    
    def TwoSampleLoss(self, Pred, Targets): # Multi-variate differentiable f-divergence
        SumDic = {}
        TwoSamples = torch.cat((Pred, Targets)).view(-1, Targets.shape[-1])
        ZeroLabels = np.zeros((len(Pred), 1))
        OneLabels = np.ones((len(Targets), 1))
        Labels = np.vstack((ZeroLabels, OneLabels))
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

            TSLoss = self.TwoSampleLoss(Pred_out2[:, :-1], targets[:, :-1])
            MSEloss = criterion(squeeze_Pred_out2[:, :-1], targets[:, :-1])

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

            TSloss = self.TwoSampleLoss(Pred_out2[:, :-1], targets[:, :-1])

            MSEloss = criterion(Pred_out2[:, :-1], targets[:, :-1])

            TotalLoss = MSEloss + self.weight*(TSloss - self.delta)**2
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
        loss = criterion(outputs2[:, :-1], targets[:, :-1])
        if args.LossType == 'L1':
            l1_parameters = []
            for parameter in model.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = args.delta * torch.abs(torch.cat(l1_parameters)).sum()
            loss+=l1           
       
        loss.backward()
        optimizer.step()
    return loss.item()

def test(args, model, device, test_loader, scaler):
    model.eval()

    Pred = np.zeros((0, test_loader.dataset.Labels.shape[-1]-1))
    TrueLabels = np.zeros((0, test_loader.dataset.Labels.shape[-1]-1));
      
    with torch.no_grad():
        for (Feats, targets) in test_loader:
            Feats, targets = Feats.to(device), targets.to(device)
            Output1, Output2 = model(Feats)
            Output2 = Output2.squeeze()

            Pred = np.vstack((Pred, Output2[:, :-1].cpu().detach().numpy()))
            TrueLabels = np.vstack((TrueLabels, targets[:, :-1].cpu().detach().numpy()))

    Pred = Pred * scaler.scale_ + scaler.mean_
        
    Err1 = mean_squared_error(Pred, TrueLabels)

    Err2 = FDivergence(Pred, TrueLabels)
    return Pred, Err1, Err2

def test2(args, model, device, test_loader, scaler, SampleWise):
    model.eval()

    if args.DataType == 'ChemCam' or args.DataType == 'ChemCam2':
        Pred = np.zeros((0, 9))
    elif args.DataType == 'Dyar_SuperCam_Mars' or args.DataType == 'Dyar_SuperCam_Earth' or args.DataType == 'Dyar_SuperCam_Vacuum':
        Pred = np.zeros((0, 10))
      
    with torch.no_grad():
        for (Feats, targets) in test_loader:
            Feats, targets = Feats.to(device), targets.to(device)
            Output1, Output2 = model(Feats)
            Output2 = Output2.squeeze()

            Pred = np.vstack((Pred, Output2[:, :-1].cpu().detach().numpy()))

    if scaler != None:
        Pred = Pred * scaler.scale_ + scaler.mean_
        
    return Pred
    
# Create a dataset class for the One-data training
class MyAIDataset(Dataset):
    def __init__(self, Feats, Labels):
        self.FeatLen = Feats.shape[-1]
        self.Feats = np.float32(Feats).reshape((-1, 1, self.FeatLen))
   
        self.Labels = np.zeros((len(Labels), Labels.shape[1] + 1))
        self.Labels[:, :-1] = Labels 
        self.Labels[:, -1] = 1 - np.sum(self.Labels, axis = 1)
                               
        self.Labels = np.float32(self.Labels) 
    
    def __len__(self):
        return len(self.Feats)

    def __getitem__(self, idx):
        return self.Feats[idx], self.Labels[idx]
